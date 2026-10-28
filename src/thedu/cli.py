"""The one CLI: `uv run thedu <command>`."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import typer

from thedu.paths import QRELS_DIR, ROOT

app = typer.Typer(add_completion=False, no_args_is_help=True, help="thedu: query-time rewriting over BM25 vs embeddings.")
analyze_app = typer.Typer(no_args_is_help=True, help="Where does each method win? Reads finished runs; retrieves and calls nothing.")
app.add_typer(analyze_app, name="analyze")

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
        run, m = run_dev(cfg, offline=offline, restart=restart, probe=not offline)
    except (GatewayError, VersionMismatch, RunRefused) as e:
        typer.echo(f"\n{e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"\n{cfg.dataset}: {m['n_questions']} dev questions   run {run.path.name}")
    typer.echo(f"{'setup':<6}{'nDCG@10':>9}  {'95% range':<18}{'Recall@100':>11}{'LLM $/search':>14}{'rerank $/search':>17}{'latency par/seq':>17}")
    soft = False
    for s, v in m["setups"].items():
        nd, rc = v["ndcg10"], v["recall100"]
        c = v["llm_cost_per_search_usd"]
        cost = "n/a" if c is None else f"{'~' if v['llm_cost_is_estimate'] else ''}{c:.5f}"
        lat = v["latency_relative_to_A0"]
        lat = "n/a" if lat is None else f"{lat['parallel']:.0f}x / {lat['sequential']:.0f}x"
        flag = "*" if v["recall100_is_soft"] else " "
        soft = soft or v["recall100_is_soft"]
        rr = v.get("rerank_cost_per_search_usd")
        rr = "" if rr is None else f"{rr:.5f}"
        typer.echo(f"{s:<6}{nd['mean']:>9.4f}  [{nd['lo']:.4f}, {nd['hi']:.4f}]{rc['mean']:>10.4f}{flag}{cost:>14}{rr:>17}{lat:>17}")
    if soft:
        typer.echo("* soft: many questions match fewer than 100 documents, so the rest of this BM25 top 100 is arbitrary filler.")
    if any(v["llm_cost_is_estimate"] for v in m["setups"].values()):
        typer.echo("~ estimated offline from cached rewrites (hidden thinking tokens not counted).")
    typer.echo("\nPaired nDCG@10 difference (second minus first, same questions, 95% range):")
    for name, p in m["paired"].items():
        role = f"  [{p['role']}]" if "role" in p else ""
        typer.echo(f"  {name}: {p['ndcg10_diff']:+.4f}  [{p['lo']:+.4f}, {p['hi']:+.4f}]  wins {p['wins']}, losses {p['losses']}{role}")
    for name, e in m["extras"].items():
        if isinstance(e, dict):
            typer.echo(f"  {name}: {e['rescued']} of {e['a4_misses']} A4 misses rescued; "
                       f"{e['pushed_down']} of {e['a4_correct_at_1']} A4 correct-at-#1 pushed down")
        else:
            typer.echo(f"  {name}: {e:.4f}")
    typer.echo(f"\nSaved to {run.path}.")


@app.command()
def embed(dataset: str) -> None:
    """Embed a dataset's documents with the default embedding model and cache them (resumable, saved in chunks)."""
    _logging()
    from thedu.config import DenseConfig
    from thedu.data import DATASETS, read_docs
    from thedu.retrievers import DenseRetriever

    if dataset not in DATASETS:
        typer.echo(f"unknown dataset {dataset!r}; choose from {DATASETS}", err=True)
        raise typer.Exit(1)
    docs, cfg = read_docs(dataset), DenseConfig()
    DenseRetriever.from_model(dataset, list(docs), list(docs.values()), cfg.model, cfg.max_seq_length)
    typer.echo(f"{dataset}: {len(docs)} documents embedded and cached.")


@app.command()
def parity(restart: bool = typer.Option(False, help="Set an existing parity run aside and start again.")) -> None:
    """Frozen baselines (A0, A4) vs published scores on test; passes within the tolerance in tests/expected.yaml."""
    _logging()
    from thedu.parity import run_parity
    from thedu.registry import VersionMismatch

    try:
        ok, rows = run_parity(restart=restart)
    except (VersionMismatch, RuntimeError) as e:
        typer.echo(f"\n{e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"\n{'dataset':<10}{'setup':<6}{'ours':>9}{'published':>11}{'diff':>9}  run")
    for r in rows:
        typer.echo(f"{r.dataset:<10}{r.setup:<6}{r.ours:>9.4f}{r.published:>11.4f}{r.diff:>+9.4f}  {r.run_hash}")
    typer.echo("\nPARITY PASS" if ok else "\nPARITY FAIL")
    raise typer.Exit(0 if ok else 1)


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


def read_manifest() -> dict[str, str]:
    stored = {}
    for ln in MANIFEST.read_text().splitlines():
        digest, path = ln.split("  ", 1)
        stored[path] = digest
    return stored


@app.command()
def freeze(add: list[Path] = typer.Option(None, "--add", help="Append this file to the manifest (repeatable).")) -> None:
    """Make files read-only and store their checksums. The manifest only ever grows.

    With no options, freezes the judge, answer keys and golden files (once). With --add PATH,
    appends the named files; existing lines are never changed or removed.
    """
    if not add:
        if MANIFEST.exists():
            typer.echo(f"{MANIFEST.relative_to(ROOT)} already exists. Use --add PATH to freeze one more file; "
                       "frozen files are never re-frozen, and if one is wrong, stop and report.", err=True)
            raise typer.Exit(1)
        digests = frozen_digests()
        MANIFEST.write_text("".join(f"{h}  {p}\n" for p, h in digests.items()))
        for p in _frozen_files() + [MANIFEST]:
            p.chmod(0o444)
        typer.echo(f"Froze {len(digests)} files; checksums in {MANIFEST.relative_to(ROOT)}.")
        return
    if not MANIFEST.exists():
        typer.echo("Nothing is frozen yet: run `thedu freeze` first.", err=True)
        raise typer.Exit(1)
    stored = read_manifest()
    new: dict[str, str] = {}
    for p in add:
        path = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
        if not path.is_file() or ROOT not in path.parents:
            typer.echo(f"{p}: not a file inside the repo.", err=True)
            raise typer.Exit(1)
        rel = str(path.relative_to(ROOT))
        if rel in stored or rel in new:
            typer.echo(f"{rel} is already frozen; frozen lines are never changed.", err=True)
            raise typer.Exit(1)
        new[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    MANIFEST.chmod(0o644)
    with open(MANIFEST, "a") as f:
        f.writelines(f"{h}  {p}\n" for p, h in new.items())
    MANIFEST.chmod(0o444)
    for rel in new:
        (ROOT / rel).chmod(0o444)
    typer.echo(f"Added {len(new)} file(s) to {MANIFEST.relative_to(ROOT)}: {', '.join(new)}. Log this in JOURNAL.md.")


def _ci(x: dict | None) -> str:
    return "n/a" if x is None else f"{x['mean']:+.3f} [{x['lo']:+.3f},{x['hi']:+.3f}]"


@analyze_app.command("explore")
def analyze_explore(run: str = typer.Option(..., help="The SciFact M3 run id (SciFact only).")) -> None:
    """Win rates and A3-A4 difference by feature tercile, on SciFact dev."""
    from thedu.analysis.steps import explore

    r = explore(run)
    typer.echo(f"scifact dev, {r['n']} questions. Labels: {r['label_counts']}. A3-A4 overall {_ci(r['a3_minus_a4'])}")
    for f, t in r["tables"].items():
        th = r["thresholds"][f]
        typer.echo(f"\n{f}   (cut points {th[0]:.3f}, {th[1]:.3f})")
        typer.echo(f"  {'bucket':<7}{'n':>5}{'rewrite':>9}{'embed':>8}{'tie':>7}{'fail':>7}   A3-A4 mean [95% range]      A5-A4")
        for b, v in t.items():
            typer.echo(f"  {b:<7}{v['n']:>5}{v['share_rewrite-wins']:>9.0%}{v['share_embed-wins']:>8.0%}{v['share_tie']:>7.0%}"
                       f"{v['share_both-fail']:>7.0%}   {_ci(v['a3_minus_a4']):<27}{_ci(v['a5_minus_a4'])}")


@analyze_app.command("stability")
def analyze_stability(original: str = typer.Option(..., help="The SciFact M3 run id."),
                      regen: list[str] = typer.Option(..., help="Run ids of the regenerated A3 runs (repeat).")) -> None:
    """How many labels survive regenerating the rewrites."""
    from thedu.analysis.steps import stability

    r = stability(original, regen)
    typer.echo(f"{r['n']} questions; A4 mean {r['a4_mean']:.4f}")
    for k, v in r["sources"].items():
        typer.echo(f"  {k}: A3 mean {v['a3_mean']:.4f}  labels {v['label_counts']}  providers {v['providers']}")
    for k, v in r["pairs"].items():
        typer.echo(f"\n{k}\n  " + "\n  ".join(f"{a}: {b}" for a, b in v.items()))


@analyze_app.command("confirm")
def analyze_confirm(run: str = typer.Option(..., help="The M3 run id of the dataset."),
                    rules: str = typer.Option("configs/m3b_rules.yaml"),
                    dataset: str = typer.Option("nfcorpus", help="nfcorpus (the pre-registered test) or fiqa (secondary).")) -> None:
    """Test the frozen rules, unchanged, on a dev set."""
    from thedu.analysis.steps import confirm

    try:
        r = confirm(run, rules, dataset)
    except RuntimeError as e:
        typer.echo(f"\n{e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{dataset} dev, {r['n']} questions. Labels: {r['label_counts']}")
    for name, v in r["rules"].items():
        e = v["a3_minus_a4_in_vs_out"]
        typer.echo(f"\n{name}: {v['rule']}   {'HELD' if v['held'] else 'did not hold'}")
        typer.echo(f"  group {e['n_group']} questions ({e['share']:.0%}); A3-A4 in vs out: {e['effect']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]")
        a = v["a5_minus_a4_in_vs_out"]
        typer.echo(f"  A5-A4 in vs out: {a['effect']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}]")


@analyze_app.command("route")
def analyze_route(scifact_run: str = typer.Option(...), nfcorpus_run: str = typer.Option(...)) -> None:
    """A router fit on SciFact pre-search features, evaluated on NFCorpus (only if a rule held)."""
    from thedu.analysis.steps import route

    try:
        r = route(scifact_run, nfcorpus_run)
    except RuntimeError as e:
        typer.echo(f"\n{e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"features allowed (held on NFCorpus): {r['allowed_features']}")
    typer.echo(f"NFCorpus nDCG@10: {r['reference_mean_ndcg10']}; ceilings A4/A5 {r['ceiling_best_of_A4_A5']:.4f}, A3/A4 {r['ceiling_best_of_A3_A4']:.4f}")
    for ver, v in r["versions"].items():
        s = v["stump_fit_on_scifact"]
        typer.echo(f"\n{ver}: fire when {s['feature']} {s['op']} {s['threshold']}")
        typer.echo(f"  SciFact: fires on {v['scifact_share_fired']:.0%}; cross-validated gain vs always-free {v['scifact_cv_mean_gain_vs_always_free']:+.4f} "
                   f"(always-paid {v['scifact_always_paid_mean_gain']:+.4f})")
        typer.echo(f"  NFCorpus: nDCG@10 {_ci(v['nfcorpus_ndcg10'])}; calls the LLM on {v['nfcorpus_share_calling_llm']:.0%}; "
                   f"LLM $/search {v['llm_cost_per_search_usd']:.6f}{'~' if v['llm_cost_is_estimate'] else ''}")
        for k, p in v["paired"].items():
            typer.echo(f"    {k}: {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] wins {p['wins']}, losses {p['losses']}")


@analyze_app.command("router")
def analyze_router(run: str = typer.Option(..., help="The M3-style run id with A3, A4, A5 and H0."),
                   dataset: str = typer.Option(...), rules: str = typer.Option("configs/router.yaml")) -> None:
    """Evaluate the frozen router (no fitting) on a dev set."""
    from thedu.analysis.steps import router_eval

    try:
        r = router_eval(run, rules, dataset)
    except RuntimeError as e:
        typer.echo(f"\n{e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{dataset} dev, {r['n']} questions. Router: fire A5 when {r['rule']}, else A4")
    typer.echo(f"nDCG@10  A4 {r['reference_mean_ndcg10']['A4']:.4f}  H0 {r['reference_mean_ndcg10']['H0']:.4f}  "
               f"A5 {r['reference_mean_ndcg10']['A5']:.4f}  A3 {r['reference_mean_ndcg10']['A3']:.4f}  "
               f"ceiling(A4,A5) {r['ceiling_best_of_A4_A5']:.4f}")
    typer.echo(f"router   {_ci(r['router_ndcg10'])}; calls the LLM on {r['n_calling_llm']} questions ({r['share_calling_llm']:.0%}); "
               f"rewrite $/search {r['llm_cost_per_search_usd']:.6f} vs A5 {r['a5_llm_cost_per_search_usd']:.6f}"
               f"{' (estimated)' if r['llm_cost_is_estimate'] else ''}")
    for k, p in r["paired"].items():
        typer.echo(f"  router vs {k}: {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] wins {p['wins']}, losses {p['losses']}")
    typer.echo(f"\n{'SUCCESS' if r['success'] else 'NOT MET'}: router minus A4, lower bound {r['paired']['A4']['lo']:+.4f} "
               f"(needs > {r['criterion']['paired_vs_free_lower_bound_above']})")


@analyze_app.command("diagnose")
def analyze_diagnose(run: str = typer.Option(..., help="A run with A0, A3 and A4 (the M3 setups)."),
                     dataset: str = typer.Option(...)) -> None:
    """Where the relevant documents sit in the top 100, and the perfect-reranker ceiling. No LLM calls."""
    import json

    from thedu import data
    from thedu.analysis.diagnosis import diagnose, read_trec
    from thedu.analysis.steps import OUT
    from thedu.paths import ROOT

    _, queries, qrels = data.load(dataset, "dev")
    qids = sorted(queries)
    rankings = {s: read_trec(ROOT / "runs" / run / f"{s}.trec") for s in ("A0", "A3", "A4")}
    r = diagnose(rankings, qrels, qids)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"diagnosis-{dataset}.json").write_text(json.dumps({"run": run, "dataset": dataset, **r}, indent=2))
    typer.echo(f"{dataset} dev, {r['n_questions']} questions (run {run}). Best-placed relevant document:")
    typer.echo(f"  {'setup':<6}{'top 10':>8}{'11-100':>8}{'missing':>9}{'nDCG@10':>9}{'perfect rerank':>22}{'excl. zero-score':>18}")
    for s, v in r["setups"].items():
        c = v["perfect_rerank_ndcg10"]
        typer.echo(f"  {s:<6}{v['share']['top10']:>8.1%}{v['share']['ranks11-100']:>8.1%}{v['share']['missing']:>9.1%}"
                   f"{v['ndcg10']:>9.4f}{c['mean']:>10.4f} [{c['lo']:.3f},{c['hi']:.3f}]{v['perfect_rerank_ndcg10_excluding_zero_score']:>12.4f}")
    for s, p in r["paired_ceiling_vs_a4"].items():
        typer.echo(f"  perfect rerank of {s} minus A4 (actual): {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]")


sweep_app = typer.Typer(no_args_is_help=True, help="Keyword-index sweep, RM3, embedding-filtered feedback (dev only, no LLM calls).")
app.add_typer(sweep_app, name="sweep")


def _sweep(name: str, dataset: str | None, fn_name: str) -> None:
    import thedu.sweeps as sw

    _logging()
    for d in ([dataset] if dataset else sw.DATASETS):
        typer.echo(f"{name} on {d}: saved {getattr(sw, fn_name)(d)}")
    typer.echo(f"\nKeep rule: paired nDCG@10 range above zero (lower bound > 0) on at least two of three datasets.")
    for label, e in sw.keep_verdicts(name).items():
        per = "  ".join(f"{d[:4]} {p['diff']:+.4f}[{p['lo']:+.4f},{p['hi']:+.4f}]" for d, p in e["datasets"].items())
        typer.echo(f"  {'KEEP' if e['kept'] else 'drop'}  {label:<28}{per}")


@sweep_app.command("index")
def sweep_index(dataset: str = typer.Option(None, help="One dataset; default all three.")) -> None:
    """The keyword-index sweep, one change at a time against the current A0."""
    _sweep("index", dataset, "index_sweep")


@sweep_app.command("rm3")
def sweep_rm3(dataset: str = typer.Option(None)) -> None:
    """RM3 pseudo-relevance feedback (no model), vs A0, A3 and A4."""
    _sweep("rm3", dataset, "rm3_sweep")


@sweep_app.command("efb")
def sweep_efb(dataset: str = typer.Option(None)) -> None:
    """Embedding-filtered feedback (an embedding model at query time only, as a term filter)."""
    _sweep("efb", dataset, "efb_sweep")
