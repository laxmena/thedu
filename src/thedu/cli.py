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
    for s_, v in m["setups"].items():
        if "listwise_fallbacks" in v:
            f = v["listwise_fallbacks"]
            typer.echo(f"  {s_} listwise fallbacks: {f['questions_with_fallback']} of {f['questions']} questions "
                       f"(missing ids {f['ids_missing']}, repeated {f['ids_repeated']}, invalid {f['ids_invalid']}, failed calls {f['failed_calls']})")
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


@app.command(name="data-fetch")
def data_fetch(dataset: str) -> None:
    """Download IndicQA Tamil at the pinned revision (run as `uv run --with pyarrow thedu data-fetch indicqa_ta`)."""
    from thedu.data import indicqa

    if dataset != indicqa.NAME:
        typer.echo(f"only {indicqa.NAME} is fetched this way; the BEIR sets download themselves", err=True)
        raise typer.Exit(1)
    indicqa.fetch()
    typer.echo(f"{dataset}: written to data/{dataset}/ from {indicqa.REPO}@{indicqa.REVISION}")


@app.command(name="data-split")
def data_split() -> None:
    """Write configs/indicqa_ta_split.json once (seed 7, by passage). Freeze it afterwards with `thedu freeze --add`."""
    from thedu.data import indicqa

    try:
        doc = indicqa.write_split()
    except FileExistsError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(f"passages {doc['passages']}, questions {doc['questions']}")


@app.command(name="data-stats")
def data_stats() -> None:
    """Counts and lengths of IndicQA Tamil (whole pool, no run) to runs/analysis/indicqa_ta-stats.json."""
    import json

    from thedu.data import indicqa
    from thedu.paths import runs_dir

    def counter(model: str):
        def count(texts: list[str]) -> list[int]:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model)
            return [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        return count

    # the reranker's tokenizer (the one the task asks for) and the embedding model's, which is what A4's 512 limit counts
    out = indicqa.stats({"bge_reranker": counter("BAAI/bge-reranker-v2-m3"), "qwen3_embedding": counter("Qwen/Qwen3-Embedding-0.6B")})
    path = runs_dir() / "analysis" / "indicqa_ta-stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    typer.echo(json.dumps(out, indent=1, ensure_ascii=False))


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


@analyze_app.command("depth")
def analyze_depth(run: str = typer.Option(..., help="A run with A0, A3 and A4."), dataset: str = typer.Option(...)) -> None:
    """Relevant documents at ranks 51-100, and the perfect-rerank ceiling of the top 50 vs the top 100. No runs."""
    import json

    from thedu import data
    from thedu.analysis.diagnosis import depth_table, read_trec
    from thedu.analysis.steps import OUT
    from thedu.paths import ROOT

    _, queries, qrels = data.load(dataset, "dev")
    qids = sorted(queries)
    r = depth_table({s: read_trec(ROOT / "runs" / run / f"{s}.trec") for s in ("A0", "A3")}, qrels, qids)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"depth-{dataset}.json").write_text(json.dumps({"run": run, "dataset": dataset, "n_questions": len(qids), **r}, indent=2))
    typer.echo(f"{dataset} dev, {len(qids)} questions (run {run})")
    for s, v in r.items():
        typer.echo(f"  {s}: relevant at ranks 51-100: {v['relevant_at_51_100']} (top 50: {v['relevant_in_top50']} of {v['relevant_total']} in all); "
                   f"questions with any there {v['share_questions_with_any_at_51_100']:.1%}, only there {v['share_questions_relevant_only_at_51_100']:.1%}; "
                   f"ceiling top 50 {v['ceiling_top50']:.4f}, top 100 {v['ceiling_top100']:.4f}")


m5_app = typer.Typer(no_args_is_help=True, help="M5, the made-up names test (docs/m5_codenames.md).")
app.add_typer(m5_app, name="m5")


@m5_app.command("entities")
def m5_entities() -> None:
    """Write the entity check list (configs/m5_entities.csv) for review. Uses the pinned model through the gateway."""
    _logging()
    from thedu.m5.entities import build

    typer.echo(f"wrote {build()}")


@m5_app.command("run")
def m5_run(level: int = typer.Option(..., help="0, 25, 50 or 100"), condition: str = typer.Option("A", help="A or B (level 0: A only)"),
           restart: bool = typer.Option(False)) -> None:
    """One swap level and condition of the made-up names test; saves runs/<hash>/."""
    _logging()
    from thedu.m5.runner import run_level

    run, metrics = run_level(level, condition, restart)
    typer.echo(f"level {level} condition {condition}: run {run.path.name}")
    for s, v in metrics["setups"].items():
        typer.echo(f"  {s:<5} nDCG@10 {v['ndcg10']['mean']:.4f}")


@m5_app.command("analyze")
def m5_analyze() -> None:
    """The pre-registered M5 tests (P1 to P3, S1 to S4) from the finished runs; writes runs/analysis/m5-results.json."""
    from thedu.m5.analysis import analyze

    r = analyze()
    typer.echo(f"{r['n_questions']} questions; runs {r['runs']}")
    for k, v in r["ndcg10"].items():
        typer.echo(f"  {k:<6} " + "  ".join(f"{a} {x:.4f}" for a, x in v.items()))
    for cond, v in r["primary"].items():
        typer.echo(f"condition {cond} at 100%:")
        for name, t in v["tests"].items():
            typer.echo(f"  {name:<12} {t['diff']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}]  {t['verdict']}")
        typer.echo(f"  P1 {v['P1_any_free_arm_matches_or_beats_A4']}  P2 {v['P2_A0R_vs_A4R']}  P3 {v['P3_A3R_vs_A4R']}")
    typer.echo(f"expectation holds (any of P1-P3 in both conditions): {r['expectation_holds']}")


@m5_app.command("preview")
def m5_preview() -> None:
    """Show the swap on 5 questions (both conditions, with a document excerpt), the counts and the residual audit."""
    from thedu.m5.preview import preview

    r = preview()
    typer.echo(f"{r['entities']} entities; {r['affected_questions']} of {r['dev_questions']} dev questions contain at least one; split {r['entity_split']}")
    for lv, v in r["levels"].items():
        a = v["audit"]
        typer.echo(f"level {lv}%: {v['entities']} entities, {v['documents_changed']} documents changed, {v['glossary_documents']} glossary docs, "
                   f"{v['affected_questions_changed']} affected questions changed; residual {a['loose_after']} of {a['loose_before']} ({a['share_left']:.1%})")
    typer.echo(f"relevant-passage split of the affected questions: {r['relevant_passage_split']}")
    typer.echo(f"leftovers at 100% (top 20): {r['levels'][100]['audit']['leftover']}")
    typer.echo(f"not-the-entity words set apart at 25/50/100%: " + ", ".join(str(r['levels'][lv]['audit']['excluded_not_the_entity']) for lv in (25, 50, 100)))
    for i, p in enumerate(r["pairs"], 1):
        typer.echo(f"\n--- pair {i} (question {p['qid']}; entities: {', '.join(p['entities'])})")
        typer.echo(f"original: {p['original']}\nA (codename): {p['condition_A']}\nB (description): {p['condition_B']}")
        if p["document_excerpt"]:
            typer.echo(f"relevant doc, original: ...{p['document_excerpt']['original']}...\nrelevant doc, swapped:  ...{p['document_excerpt']['swapped']}...")
        else:
            typer.echo("relevant doc: no entity mention in it")
        for g in p["glossary"]:
            typer.echo(f"glossary: {g}")


@m5_app.command("samples")
def m5_samples(entity: list[str] = typer.Option(..., "--entity", help="An entity name from the list (repeatable)."),
               n: int = typer.Option(10, "-n")) -> None:
    """In-context matches of an entity, before and after the swap (condition A), to check for common-word false matches."""
    from thedu.m5.preview import samples

    for name, rows in samples(entity, n).items():
        typer.echo(f"\n=== {name}")
        for r in rows:
            if r.get("not_an_entity"):
                typer.echo(f"not in the entity list, left alone; the word occurs {r['occurrences_in_dev_corpus_and_questions']} times")
            else:
                typer.echo(f"[{r['matched']}] ...{r['before']}...\n      -> ...{r['after']}...")


@analyze_app.command("recall50")
def analyze_recall50(run: str = typer.Option(..., help="A run with A0, A3 and A4."), dataset: str = typer.Option(...)) -> None:
    """Recall@50 of A0, A3 and A4, with A4's paired difference. No runs."""
    import json

    from thedu import data
    from thedu.analysis.diagnosis import read_trec, recall_at
    from thedu.analysis.steps import OUT
    from thedu.paths import ROOT

    _, queries, qrels = data.load(dataset, "dev")
    qids = sorted(queries)
    r = recall_at({s: read_trec(ROOT / "runs" / run / f"{s}.trec") for s in ("A0", "A3", "A4")}, qrels, qids, 50)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"recall50-{dataset}.json").write_text(json.dumps({"run": run, "dataset": dataset, "n_questions": len(qids), **r}, indent=2))
    typer.echo(f"{dataset} dev, {len(qids)} questions (run {run}): Recall@50")
    for s, v in r.items():
        extra = "" if "a4_minus_this" not in v else f"   A4 minus {s}: {v['a4_minus_this']['diff']:+.4f} [{v['a4_minus_this']['lo']:+.4f}, {v['a4_minus_this']['hi']:+.4f}]"
        typer.echo(f"  {s}: {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]{extra}")


@analyze_app.command("paired")
def analyze_paired(a: str = typer.Option(..., help="RUN:SETUP, the base, e.g. 00fb868eaa:A4R"),
                   b: str = typer.Option(..., help="RUN:SETUP, compared with the base"),
                   label: str = typer.Option("", help="A name for the saved file.")) -> None:
    """Paired nDCG@10 of b minus a over the same questions, from two finished runs. Retrieves nothing."""
    import json

    from thedu.analysis.steps import OUT, load_rows
    from thedu.judge import paired_difference

    (ra, sa), (rb, sb) = a.split(":"), b.split(":")
    xa, xb = load_rows(ra)[sa], load_rows(rb)[sb]
    qids = sorted(set(xa) & set(xb))
    if len(qids) != len(xa) or len(qids) != len(xb):
        raise typer.BadParameter(f"the runs cover different questions ({len(xa)} vs {len(xb)}, shared {len(qids)})")
    p = paired_difference([xa[q]["ndcg10"] for q in qids], [xb[q]["ndcg10"] for q in qids])
    res = {"base": a, "other": b, "n": len(qids), "diff": p.mean, "lo": p.lo, "hi": p.hi, "wins": p.wins, "losses": p.losses}
    if label:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"paired-{label}.json").write_text(json.dumps(res, indent=2))
    typer.echo(f"{b} minus {a}, {len(qids)} questions: {p.mean:+.4f} [{p.lo:+.4f}, {p.hi:+.4f}] wins {p.wins}, losses {p.losses}")


@analyze_app.command("llm-rerank")
def analyze_llm_rerank(dataset: str = typer.Option(...), sample_run: str = typer.Option(..., help="The 100-question run with A0, A3, A4, A0L, A3L, A4L."),
                       bge_run: str = typer.Option(..., help="The full run with A0R and A3R."),
                       a4r_run: str = typer.Option(..., help="The full run with A4R.")) -> None:
    """LLM vs bge reranking on the fixed sample: nDCG@10, headroom captured, paired tests, cost and latency."""
    import json

    import numpy as np

    from thedu import data
    from thedu.analysis.diagnosis import ceiling_top, read_trec
    from thedu.analysis.steps import OUT, load_rows
    from thedu.judge import paired_difference
    from thedu.paths import ROOT
    from thedu.report import _latencies

    _, _, qrels = data.load(dataset, "dev")
    s, b, c = load_rows(sample_run), load_rows(bge_run), load_rows(a4r_run)
    qids = sorted(s["A0"])
    arm = {n: np.array([s[n][q]["ndcg10"] for q in qids]) for n in ("A0", "A3", "A4", "A0L", "A3L", "A4L")}
    arm["A0R"], arm["A3R"] = (np.array([b[n][q]["ndcg10"] for q in qids]) for n in ("A0R", "A3R"))
    arm["A4R"] = np.array([c["A4R"][q]["ndcg10"] for q in qids])
    ceil = {n: float(np.mean([ceiling_top(read_trec(ROOT / "runs" / sample_run / f"{n}.trec")[q], qrels[q]) for q in qids])) for n in ("A0", "A3", "A4")}
    out: dict = {"dataset": dataset, "n": len(qids), "ndcg10": {k: float(v.mean()) for k, v in arm.items()}, "ceiling_top50": ceil,
                 "headroom_captured": {}, "paired": {}, "cost_and_latency": {}}
    for base in ("A0", "A3", "A4"):
        for kind in ("R", "L"):
            n = f"{base}{kind}"
            out["headroom_captured"][n] = float((arm[n].mean() - arm[base].mean()) / (ceil[base] - arm[base].mean()))

    def pair(x: str, y: str, role: str) -> None:
        p = paired_difference(arm[x], arm[y])
        out["paired"][f"{y} vs {x}"] = {"diff": p.mean, "lo": p.lo, "hi": p.hi, "wins": p.wins, "losses": p.losses, "role": role}

    pair("A3R", "A3L", "primary")
    for x, y in (("A0R", "A0L"), ("A4L", "A3L"), ("A4L", "A0L"), ("A4R", "A3L"), ("A4R", "A0L"), ("A4", "A3L"), ("A4", "A0L"),
                 ("A4", "A4L"), ("A4R", "A4L"), ("A3", "A3L")):
        pair(x, y, "secondary")
    # Cost per search (rewrite + rerank at reference rates; embedding search is unpriced for all A4 arms) and mean time per search
    # in the runs. The cross-encoder runs shared the machine with other jobs, so their times are inflated; see the latency note.
    ms = lambda rows, name: float(np.mean([r["rank_ms"] + sum(p["retrieval_ms"] + p["llm_ms"] for p in r["parts"].values()) for r in (rows[name][q] for q in qids)]))
    for n, rows in (("A0L", s), ("A3L", s), ("A4L", s), ("A0R", b), ("A3R", b), ("A4R", c)):
        m = json.loads((ROOT / "runs" / (sample_run if rows is s else bge_run if rows is b else a4r_run) / "metrics.json").read_text())["setups"][n]
        out["cost_and_latency"][n] = {"llm_cost_per_search_usd": m["llm_cost_per_search_usd"], "rerank_cost_per_search_usd": m.get("rerank_cost_per_search_usd"),
                                      "mean_ms_in_run": ms(rows, n)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"llm-rerank-{dataset}.json").write_text(json.dumps(out, indent=2))
    typer.echo(f"{dataset}: {len(qids)} sample questions   nDCG@10: " + "  ".join(f"{k} {v:.4f}" for k, v in out["ndcg10"].items()))
    typer.echo(f"  top-50 ceiling: " + "  ".join(f"{k} {v:.4f}" for k, v in ceil.items()) + "   headroom captured: " +
               "  ".join(f"{k} {v:.0%}" for k, v in out["headroom_captured"].items()))
    for k, p in out["paired"].items():
        typer.echo(f"  {k}: {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] wins {p['wins']}, losses {p['losses']}  [{p['role']}]")
    typer.echo("  cost per search (rewrite+rerank; embedding unpriced) and mean ms per search in the run:")
    for n, v in out["cost_and_latency"].items():
        cost = (v["llm_cost_per_search_usd"] or 0) + (v["rerank_cost_per_search_usd"] or 0)
        typer.echo(f"    {n}: ${cost:.5f}   {v['mean_ms_in_run'] / 1000:.2f} s")


@analyze_app.command("retime")
def analyze_retime(dataset: str = typer.Option(...), sample_run: str = typer.Option(..., help="The 100-question run with A0, A3, A4.")) -> None:
    """Clean cross-encoder timing: rerank the sample questions' saved top-50 lists, one at a time, nothing else running.

    Uses the candidate lists of A0, A3 and A4 from the sample run. Run it on an idle machine; the per-search time is
    quoted only relative to the other setups (AGENTS.md), never as an absolute laptop timing.
    """
    import json
    import os
    import statistics
    import time

    from thedu import data
    from thedu.analysis.diagnosis import read_trec
    from thedu.analysis.steps import OUT
    from thedu.config import RerankConfig
    from thedu.paths import ROOT
    from thedu.ranker.rerank import CrossEncoderReranker

    docs, queries, _ = data.load(dataset, "dev")
    cfg = RerankConfig()
    rr = CrossEncoderReranker(docs, cfg.model, cfg.depth, cfg.max_length)
    lists = {n: read_trec(ROOT / "runs" / sample_run / f"{n}.trec") for n in ("A0", "A3", "A4")}
    qids = sorted(lists["A0"])
    first = lists["A0"][qids[0]]
    rr.rerank(first, queries[qids[0]])  # warm-up: model load and device start, not timed
    out: dict = {"dataset": dataset, "sample_run": sample_run, "n": len(qids), "model": cfg.model, "depth": cfg.depth,
                 "load_average_at_start": os.getloadavg(), "setups": {}}
    for n, lst in lists.items():
        ms, pairs = [], []
        for q in qids:
            t0 = time.perf_counter()
            _, _ = rr.rerank(lst[q], queries[q])
            ms.append((time.perf_counter() - t0) * 1000)
            pairs.append(len([h for h in lst[q] if h[1] > 0][: cfg.depth]))
        out["setups"][n + "R"] = {"mean_ms": statistics.fmean(ms), "median_ms": statistics.median(ms), "mean_pairs": statistics.fmean(pairs),
                                  "pairs_per_s": sum(pairs) / (sum(ms) / 1000)}
        typer.echo(f"{dataset} {n}R: mean {statistics.fmean(ms) / 1000:.2f} s per search ({statistics.fmean(pairs):.1f} pairs), {sum(pairs) / (sum(ms) / 1000):.1f} pairs/s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"retime-{dataset}.json").write_text(json.dumps(out, indent=2))


@analyze_app.command("retime-interleaved")
def analyze_retime_interleaved(dataset: str = typer.Option(...), sample_run: str = typer.Option(..., help="The 100-question run with A0, A3, A4."),
                               passes: int = typer.Option(2, help="Passes over the same sample, each with its own load reading.")) -> None:
    """Cross-encoder timing with the setups interleaved: for each question A0R, A3R and A4R are timed back to back
    (the order rotates by question), so every setup meets the same machine load. Reports ratios to A0R only
    (AGENTS.md: relative latency, never absolute), with the load average at the start and end of each pass."""
    import json
    import os
    import statistics
    import time

    from thedu import data
    from thedu.analysis.diagnosis import read_trec
    from thedu.analysis.steps import OUT
    from thedu.config import RerankConfig
    from thedu.paths import ROOT
    from thedu.ranker.rerank import CrossEncoderReranker

    docs, queries, _ = data.load(dataset, "dev")
    cfg = RerankConfig()
    rr = CrossEncoderReranker(docs, cfg.model, cfg.depth, cfg.max_length)
    names = ("A0", "A3", "A4")
    lists = {n: read_trec(ROOT / "runs" / sample_run / f"{n}.trec") for n in names}
    qids = sorted(lists["A0"])
    rr.rerank(lists["A0"][qids[0]], queries[qids[0]])  # warm-up, not timed
    out: dict = {"dataset": dataset, "sample_run": sample_run, "n": len(qids), "method": "interleaved per question, order rotated",
                 "passes": []}
    for p in range(passes):
        load0 = os.getloadavg()
        ms: dict[str, list[float]] = {n: [] for n in names}
        for i, q in enumerate(qids):
            for n in names[i % 3:] + names[: i % 3]:
                t0 = time.perf_counter()
                rr.rerank(lists[n][q], queries[q])
                ms[n].append((time.perf_counter() - t0) * 1000)
        mean = {n: statistics.fmean(v) for n, v in ms.items()}
        out["passes"].append({"load_start": load0, "load_end": os.getloadavg(),
                              "ratio_A3R_to_A0R": mean["A3"] / mean["A0"], "ratio_A4R_to_A0R": mean["A4"] / mean["A0"]})
        typer.echo(f"{dataset} pass {p + 1}: A3R/A0R {mean['A3'] / mean['A0']:.3f}  A4R/A0R {mean['A4'] / mean['A0']:.3f}  load {load0[0]:.2f} -> {os.getloadavg()[0]:.2f}")
    r = out["passes"]
    out["agreement"] = {k: abs(r[0][k] - r[1][k]) / r[0][k] for k in ("ratio_A3R_to_A0R", "ratio_A4R_to_A0R")} if len(r) >= 2 else None
    out["quotable"] = bool(out["agreement"] and all(v <= 0.10 for v in out["agreement"].values()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"retime-interleaved-{dataset}.json").write_text(json.dumps(out, indent=2))
    if out["agreement"]:
        typer.echo(f"{dataset}: difference between passes {out['agreement']}  quotable (within 10%): {out['quotable']}")
