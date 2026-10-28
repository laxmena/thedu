"""The one CLI: `uv run thedu <command>`."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import typer

from thedu.paths import QRELS_DIR, ROOT

app = typer.Typer(add_completion=False, no_args_is_help=True, help="thedu: query-time rewriting over BM25 vs embeddings.")

MANIFEST = ROOT / "tests" / "frozen.sha256"


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)


def _pytest(args: list[str]) -> int:
    import pytest

    return int(pytest.main(["-q", *args]))


@app.command()
def check() -> None:
    """Fast tests (unit, contract, schema, checksum, leakage, A0/A3 reproduction). Under a minute."""
    raise typer.Exit(_pytest(["-m", "not smoke"]))


@app.command()
def smoke() -> None:
    """Slower end-to-end checks: A4 reproduction on SciFact and a 20-question run. Run before `thedu dev`."""
    raise typer.Exit(_pytest(["-m", "smoke"]))


@app.command()
def dev(
    config: Path,
    restart: bool = typer.Option(False, help="Set an existing run with this hash aside and start again."),
    offline: bool = typer.Option(False, help="Fail on any LLM cache miss instead of calling the model."),
) -> None:
    """Run a config on its dev questions and save runs/<hash>/."""
    _logging()
    from thedu.config import load_config
    from thedu.llm import GatewayError
    from thedu.pipeline import RunRefused, run_dev
    from thedu.registry import VersionMismatch

    cfg = load_config(config)
    try:
        run, m = run_dev(cfg, offline=offline, restart=restart)
    except (GatewayError, VersionMismatch, RunRefused) as e:
        typer.echo(f"\n{e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"\n{cfg.dataset}: {m['n_questions']} dev questions   run {run.path.name}")
    typer.echo(f"{'setup':<6}{'nDCG@10':>9}  {'95% range':<18}{'Recall@100':>11}{'$/search':>10}{'latency':>9}")
    for s, v in m["setups"].items():
        nd, rc = v["ndcg10"], v["recall100"]
        cost = "n/a" if v["cost_per_search_usd"] is None else f"{v['cost_per_search_usd']:.5f}"
        lat = "n/a" if v["latency_relative_to_A0"] is None else f"{v['latency_relative_to_A0']:.1f}x"
        typer.echo(f"{s:<6}{nd['mean']:>9.4f}  [{nd['lo']:.4f}, {nd['hi']:.4f}]{rc['mean']:>11.4f}{cost:>10}{lat:>9}")
    typer.echo("\nPaired nDCG@10 difference vs A0 (same questions):")
    for s, p in m["paired_vs_A0"].items():
        typer.echo(f"  {s}: {p['ndcg10_diff']:+.4f}  [{p['lo']:+.4f}, {p['hi']:+.4f}]  wins {p['wins']}, losses {p['losses']}")
    typer.echo(f"\nSaved to {run.path}. Write {run.path.name}/NOTES.md (five lines) when you have read the result.")


@app.command()
def parity() -> None:
    """Frozen baselines vs published scores, on test (built in M1)."""
    from thedu.parity import run_parity

    run_parity()


@app.command(name="test-eval")
def test_eval() -> None:
    """The final run on test (built in M7)."""
    from thedu.test_eval import run_test_eval

    run_test_eval()


def _frozen_files() -> list[Path]:
    files = sorted((ROOT / "src" / "thedu" / "judge").glob("*.py"))
    files += sorted(p for p in QRELS_DIR.rglob("*.tsv"))
    files += sorted(p for p in (ROOT / "tests" / "golden").rglob("*") if p.is_file())
    if (expected := ROOT / "tests" / "expected.yaml").exists():
        files.append(expected)
    return files


def frozen_digests() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in _frozen_files()}


@app.command()
def freeze() -> None:
    """Make the judge, answer keys and golden files read-only and store their checksums (once)."""
    if MANIFEST.exists():
        typer.echo(f"{MANIFEST.relative_to(ROOT)} already exists. Frozen files are never re-frozen; "
                   "if one is wrong, stop and report.", err=True)
        raise typer.Exit(1)
    digests = frozen_digests()
    MANIFEST.write_text("".join(f"{h}  {p}\n" for p, h in digests.items()))
    for p in _frozen_files() + [MANIFEST]:
        p.chmod(0o444)
    typer.echo(f"Froze {len(digests)} files; checksums in {MANIFEST.relative_to(ROOT)}.")
