"""Run registry: runs/<config-hash>/ with the config, per-question scores and TREC rankings.

Everything is appended as the run goes. A TREC line is written before its score row, and the
score row is the commit marker: on resume, rankings without a committed row are dropped.
A run records the code version it started with; resuming under different code is refused,
so a run never mixes two code versions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from threading import Lock

import yaml

from thedu.config import RunConfig
from thedu.paths import ROOT

log = logging.getLogger(__name__)

Hit = tuple[str, float]


class VersionMismatch(RuntimeError):
    pass


def code_version() -> dict[str, str]:
    """The git commit plus a hash of the package source (so uncommitted edits count too)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    h = hashlib.sha256()
    src = ROOT / "src" / "thedu"
    for p in sorted(src.rglob("*")):
        if p.suffix in {".py", ".txt"} and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(src)).encode())
            h.update(p.read_bytes())
    return {"git_commit": commit, "code_hash": h.hexdigest()[:12]}


class RunDir:
    def __init__(self, path: Path, rows: dict[tuple[str, str], dict]):
        self.path = path
        self.rows = rows
        self._lock = Lock()
        self._jsonl = open(path / "per_question.jsonl", "a", buffering=1)
        self._trec: dict[str, object] = {}

    # ---------- open / resume ----------

    @classmethod
    def open(cls, cfg: RunConfig, run_hash: str, root: Path, rates: dict[str, float], restart: bool = False,
             kind: str = "dev") -> RunDir:
        path = root / run_hash
        version = code_version()
        if path.exists():
            meta = json.loads((path / "meta.json").read_text())
            old = {k: meta.get(k) for k in version}
            if old != version:
                if not restart:
                    raise VersionMismatch(
                        f"runs/{run_hash} was started with code {old['git_commit'][:10]}/{old['code_hash']}, "
                        f"but the code is now {version['git_commit'][:10]}/{version['code_hash']}. "
                        "Resuming would mix two code versions. Rerun with --restart to set the old run aside "
                        "and start this config again from scratch."
                    )
                aside = root / f"{run_hash}.superseded-{time.strftime('%Y%m%d-%H%M%S')}"
                shutil.move(path, aside)
                log.info("moved the old run to %s", aside.name)
        if not path.exists():
            path.mkdir(parents=True)
            (path / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
            meta = {"hash": run_hash, "kind": kind, "status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "rates_usd_per_mtok": rates, **version}
            (path / "meta.json").write_text(json.dumps(meta, indent=2))
        rows = cls._load_rows(path)
        cls._repair_trec(path, rows)
        return cls(path, rows)

    @staticmethod
    def _load_rows(path: Path) -> dict[tuple[str, str], dict]:
        rows: dict[tuple[str, str], dict] = {}
        f = path / "per_question.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # torn last line from a crash: that question is redone
                    continue
                rows[(r["setup"], r["qid"])] = r
        return rows

    @staticmethod
    def _repair_trec(path: Path, rows: dict[tuple[str, str], dict]) -> None:
        for f in path.glob("*.trec"):
            setup = f.stem
            keep = [ln for ln in f.read_text().splitlines() if (setup, ln.split(" ", 1)[0]) in rows]
            f.write_text("".join(ln + "\n" for ln in keep))

    # ---------- writing ----------

    def append(self, setup: str, qid: str, hits: list[Hit], row: dict) -> None:
        lines = "".join(f"{qid} Q0 {d} {i + 1} {s:.8g} {setup}\n" for i, (d, s) in enumerate(hits))
        with self._lock:
            if setup not in self._trec:
                self._trec[setup] = open(self.path / f"{setup}.trec", "a", buffering=1)
            self._trec[setup].write(lines)
            self._trec[setup].flush()
            self._jsonl.write(json.dumps({"setup": setup, "qid": qid, **row}, ensure_ascii=False) + "\n")
            self._jsonl.flush()
            self.rows[(setup, qid)] = {"setup": setup, "qid": qid, **row}

    def finish(self, metrics: dict, status: str = "complete") -> None:
        meta = json.loads((self.path / "meta.json").read_text())
        meta["status"] = status
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2))
        (self.path / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def close(self) -> None:
        self._jsonl.close()
        for f in self._trec.values():
            f.close()
