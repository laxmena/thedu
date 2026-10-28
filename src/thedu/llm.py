"""The LLM gateway. Every LLM call in the project goes through `Gateway.call`.

- Each response is appended to an on-disk JSONL cache the moment it arrives.
- Tokens, reference cost and latency are recorded per call.
- Seed, temperature and model come from the config.
- The model and key are checked once, before the first uncached call.
- The cache's recorded spend is held under the budget in configs/budget.yaml.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from thedu.config import LLMConfig
from thedu.judge import call_cost
from thedu.pricing import Rate

log = logging.getLogger(__name__)


class GatewayError(RuntimeError):
    """Bad model, bad key, offline cache miss, or budget reached."""


@dataclass
class Result:
    text: str
    tokens_in: int | None = None  # None: the spike cache did not record token counts
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    cached: bool = False
    failed: bool = False


@dataclass
class Usage:
    """Totals over the calls behind one answer."""

    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    llm_ms: float = 0.0
    uncosted_calls: int = 0  # calls with no recorded tokens (spike cache): cost unknown
    untimed_calls: int = 0
    failed_calls: int = 0

    def add(self, r: Result) -> None:
        self.calls += 1
        self.failed_calls += r.failed
        if r.cost_usd is None:
            self.uncosted_calls += 1
        else:
            self.tokens_in += r.tokens_in or 0
            self.tokens_out += r.tokens_out or 0
            self.cost_usd += r.cost_usd
        if r.latency_ms is None:
            self.untimed_calls += 1
        else:
            self.llm_ms += r.latency_ms


@dataclass
class Gateway:
    cfg: LLMConfig
    rate: Rate
    cache_path: Path
    legacy_path: Path | None = None  # the spike's rewrites.json, read-only
    offline: bool = False  # a cache miss is an error (used by tests)
    budget_usd: float | None = None
    client: Any = None  # injectable for tests
    _cache: dict[str, dict] = field(default_factory=dict, init=False)
    _legacy: dict[str, str] = field(default_factory=dict, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _checked: bool = field(default=False, init=False)
    _spent: float = field(default=0.0, init=False)
    _file: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:  # a crash can leave one torn last line
                        log.warning("skipping a torn line in %s", self.cache_path)
                        continue
                    self._cache[e["key"]] = e
                    self._spent += e.get("cost_usd") or 0.0
        if self.legacy_path and self.legacy_path.exists():
            self._legacy = json.loads(self.legacy_path.read_text())

    # ---------- public ----------

    def call(self, prompt: str, *, attempt: int = 0, legacy_key: str | None = None) -> Result:
        """One cached completion. `attempt` > 0 asks again after an unusable answer.

        `legacy_key` is looked up in the spike's cache (attempt 0 only), which keyed
        entries by "<model>::<question-derived key>" and recorded text but no tokens.
        """
        c = self.cfg
        key = hashlib.sha256(
            json.dumps([c.model, c.temperature, c.seed, c.max_tokens, prompt, attempt]).encode()
        ).hexdigest()
        with self._lock:
            if key in self._cache:
                e = self._cache[key]
                return Result(e["text"], e["tokens_in"], e["tokens_out"], e["cost_usd"], e["latency_ms"], cached=True)
            if legacy_key is not None and attempt == 0 and (lk := f"{c.model}::{legacy_key}") in self._legacy:
                return Result(self._legacy[lk], cached=True)
        if self.offline:
            raise GatewayError("cache miss while offline: the cached rewrites do not cover this call")
        self._ensure_ready()
        if self.budget_usd is not None and self._spent >= self.budget_usd:
            raise GatewayError(
                f"budget reached: ${self._spent:.2f} recorded of ${self.budget_usd:.2f}. Stop and ask a human."
            )
        return self._request(key, prompt)

    # ---------- internals ----------

    def _ensure_ready(self) -> None:
        with self._lock:
            if self._checked:
                return
            if self.client is None:
                from dotenv import load_dotenv
                from openai import OpenAI

                from thedu.paths import ROOT

                load_dotenv(ROOT / ".env")
                self.client = OpenAI(base_url=self.cfg.base_url, api_key=os.environ.get(self.cfg.key_env, "not-needed"))
            from openai import AuthenticationError, NotFoundError, PermissionDeniedError

            try:
                self.client.chat.completions.create(
                    model=self.cfg.model, max_tokens=5, messages=[{"role": "user", "content": "ok"}]
                )
            except NotFoundError as e:
                raise GatewayError(
                    f"model {self.cfg.model!r} was not found at {self.cfg.base_url}. "
                    "List the models with: curl -s https://openrouter.ai/api/v1/models"
                ) from e
            except (AuthenticationError, PermissionDeniedError) as e:
                raise GatewayError(
                    f"the API key in {self.cfg.key_env} was rejected. Check .env and the key's credit limit."
                ) from e
            self._checked = True

    def _request(self, key: str, prompt: str) -> Result:
        c = self.cfg
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                r = self.client.chat.completions.create(
                    model=c.model, temperature=c.temperature, seed=c.seed, max_tokens=c.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:  # noqa: BLE001 - retry any provider error, then report a failed call
                log.warning("llm call failed (try %d/3): %s: %s", attempt + 1, type(e).__name__, str(e)[:120])
                time.sleep(2**attempt)
                continue
            latency = (time.perf_counter() - t0) * 1000
            text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
            tin = r.usage.prompt_tokens if r.usage else 0
            tout = r.usage.completion_tokens if r.usage else 0
            cost = call_cost(tin, tout, self.rate.input_per_mtok_usd, self.rate.output_per_mtok_usd)
            entry = {"key": key, "model": c.model, "text": text, "tokens_in": tin, "tokens_out": tout,
                     "cost_usd": cost, "latency_ms": latency, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with self._lock:
                self._append(entry)
                self._cache[key] = entry
                self._spent += cost
            return Result(text, tin, tout, cost, latency)
        return Result("", failed=True)

    def _append(self, entry: dict) -> None:
        if self._file is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.cache_path, "a", buffering=1)
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()
