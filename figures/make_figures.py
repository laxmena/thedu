"""Build the post's static figures from run files. Run from the repo root:
    python3 figures/make_figures.py
Writes figures/out/fig*.png (2x) and .svg. Dev-set numbers until the test run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

A = Path("runs/analysis"); OUT = Path("figures/out"); OUT.mkdir(parents=True, exist_ok=True)
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
KW, EMB = "#eb6834", "#2a78d6"   # embedding-free = orange, embeddings = blue (validated pair)
DS = [("scifact", "SciFact"), ("nfcorpus", "NFCorpus"), ("fiqa", "FiQA")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": BG,
    "axes.facecolor": BG, "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False})

def j(name): return json.loads((A / name).read_text())
def title(fig, t, s):
    fig.text(0.04, 0.965, t, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.04, 0.905, s, ha="left", va="top", fontsize=9.5, color=INK2)
def save(fig, n):
    fig.savefig(OUT / f"{n}.png", dpi=200); fig.savefig(OUT / f"{n}.svg"); plt.close(fig)

# Figure 1: the pipeline (search, optional LLM read-and-rewrite, search again, optional rerank)
fig, ax = plt.subplots(figsize=(9, 3.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
steps = [("Question", "", INK2, False), ("Search", "top 5", INK2, False), ("LLM adds\nsearch words", "optional", KW, True),
         ("Search again", "top 100", INK2, False), ("Rerank top 50", "optional", INK2, True), ("Top 10", "", INK2, False)]
x, w = 1.5, 13.4
for i, (h, sub, c, opt) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((x, 12), w, 12, boxstyle="round,pad=0,rounding_size=1.5", fc=BG, ec=c, lw=1.4, ls="--" if opt else "-"))
    ax.text(x + w / 2, 19.6 if sub else 18, h, ha="center", va="center", fontsize=8, color=INK, fontweight="bold")
    if sub: ax.text(x + w / 2, 14.6, sub, ha="center", va="center", fontsize=7.5, color=INK2)
    if i < len(steps) - 1: ax.annotate("", xy=(x + w + 2.9, 18), xytext=(x + w + 0.3, 18), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    x += w + 3.2
ax.text(1.5, 5, "Dashed boxes are optional. Embeddings replace both searches with one vector search over the top 100.\nUnder the strict rule no model touches the documents, so the only thing built ahead of time is the keyword index.", fontsize=7.8, color=INK2, va="center")
fig.text(0.02, 0.93, "One search, start to finish", fontsize=12, fontweight="bold", color=INK, va="top")
save(fig, "fig1-pipeline")

# Figure 3: candidate ceilings (dumbbells: as ranked -> perfectly reordered)
fig, ax = plt.subplots(figsize=(8, 4.2)); fig.subplots_adjust(left=0.14, right=0.96, top=0.78, bottom=0.14)
for r, (k, name) in enumerate(DS):
    s = j(f"diagnosis-{k}.json")["setups"]; y0 = len(DS) - 1 - r
    for dy, key, c, lab in [(0.15, "A0", KW, "Keyword"), (-0.15, "A4", EMB, "Embeddings")]:
        a, p = s[key]["ndcg10"], s[key]["perfect_rerank_ndcg10"]; y = y0 + dy
        ax.plot([a, p["mean"]], [y, y], color=c, lw=2, alpha=0.45, solid_capstyle="round")
        ax.plot([p["lo"], p["hi"]], [y, y], color=c, lw=1, alpha=0.9)
        ax.plot(a, y, "o", ms=8, mfc=BG, mec=c, mew=1.8)
        ax.plot(p["mean"], y, "o", ms=8, color=c, mec=BG, mew=1.5)
        ax.text(p["hi"] + 0.012, y, f"{lab}  {a:.2f} to {p['mean']:.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(DS))); ax.set_yticklabels([n for _, n in DS][::-1], color=INK)
ax.set_xlim(0.2, 1.2); ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("nDCG@10   (hollow: as ranked; filled: top 100 perfectly reordered, with 95% interval)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
title(fig, "Reordered perfectly, keyword search's top 100 would beat embeddings",
      "Ceilings use the top 100 candidates (the reranker reads the top 50). Embeddings' own top 100\nhas a higher ceiling still: the first sign the gap is in the candidates.")
save(fig, "fig3-ceilings")

# Figure 4: the fair fight, paired difference vs embeddings + reranker
fig, ax = plt.subplots(figsize=(8, 4.0)); fig.subplots_adjust(left=0.14, right=0.96, top=0.76, bottom=0.15)
ax.axvspan(-0.01, 0, color=GRID, alpha=0.7, lw=0); ax.axvline(0, color=INK2, lw=1)
ax.text(-0.005, -0.42, "match band: whole interval above -0.01", ha="center", va="bottom", fontsize=7.5, color=INK2)
for r, (k, name) in enumerate(DS):
    y0 = len(DS) - 1 - r
    for dy, arm, filled, lab in [(0.13, "A0R", True, "Keyword + reranker"), (-0.13, "A3R", False, "Keyword + rewrite + reranker")]:
        p = j(f"paired-a4r-{k}-{arm}.json"); y = y0 + dy
        ax.plot([p["lo"], p["hi"]], [y, y], color=KW, lw=1.6)
        ax.plot(p["diff"], y, "o", ms=8, mfc=KW if filled else BG, mec=KW, mew=1.8, label=lab if r == 0 else None)
        ax.text(p["lo"] - 0.004, y, f"{p['diff']:+.3f}", ha="right", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(DS))); ax.set_yticklabels([n for _, n in DS][::-1], color=INK)
ax.set_xlim(-0.14, 0.02); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("nDCG@10 difference vs embeddings + the same reranker   (left of 0 = embeddings ahead)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.legend(loc="upper left", frameon=False, fontsize=8.5); ax.set_ylim(-0.5, 2.5)
title(fig, "With the same reranker on both sides, embeddings stay ahead",
      "About one point behind on SciFact, one to two on NFCorpus, eight to nine on FiQA. Dots are paired\nmeans over every dev question; lines are 95% bootstrap intervals. No interval sits fully in the band.")
save(fig, "fig4-fair-fight")

# Figure 5: made-up names
m = j("m5-results.json")["ndcg10"]; lv = [0, 25, 50, 100]
fig, axes = plt.subplots(1, 2, figsize=(8, 4.0), sharey=True); fig.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.15, wspace=0.08)
series = [("A0", KW, ":", "Keyword"), ("A0R", KW, "-", "Keyword + reranker"), ("A4", EMB, "--", "Embeddings"), ("A4R", EMB, "-", "Embeddings + reranker")]
for ax, cond, sub in [(axes[0], "A", "Users type the codename"), (axes[1], "B", "Users describe it instead")]:
    for arm, c, ls, lab in series:
        ys = [m[f"L{l}{cond}"][arm] for l in lv]
        ax.plot(lv, ys, color=c, ls=ls, lw=2, marker="o", ms=5, mec=BG, label=lab)
    ax.set_title(sub, loc="left", fontsize=10, color=INK)
    ax.set_xticks(lv); ax.set_xticklabels([f"{l}%" for l in lv]); ax.set_xlim(-5, 105)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlabel("Names swapped for codenames")
axes[0].set_ylabel("nDCG@10"); axes[0].set_ylim(0.1, 0.6)
axes[0].legend(loc="lower center", bbox_to_anchor=(1.04, 1.09), ncol=4, frameon=False, fontsize=8.5, handlelength=3)
g0 = m["L0A"]["A4R"] - m["L0A"]["A0R"]; g1 = m["L100A"]["A4R"] - m["L100A"]["A0R"]
title(fig, "Codenames shrink the gap between reranked pipelines but don't close it",
      f"94 FiQA questions that mention one of 54 swapped names. When users type the codename, the reranked\ngap narrows from {g0:.3f} to {g1:.3f}. When they describe it, plain keyword search has little to match.")
save(fig, "fig5-codenames")
# Figure 6: rerank depth (150 FiQA questions)
import math
qr = {}
for i, l in enumerate(open("data/fiqa/qrels/dev.tsv")):
    if i == 0: continue
    a, b, sc = l.rstrip("\n").split("\t")
    if int(sc) > 0: qr.setdefault(a, {})[b] = int(sc)
def nd(r, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(r[:10]))
    return dcg / sum(g / math.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
rows = [json.loads(l) for l in open(A / "h2b-scores.jsonl")]
sw = j("h2b-depth-sweep.json"); D = [50, 100, 200, 500]
kw = [sum(nd([x for _, x in sorted(zip(r["a0_scores"][:d], r["a0_candidates"][:d]), key=lambda t: -t[0])], qr[r["qid"]]) for r in rows) / len(rows) for d in D]
a4 = sum(nd(r["a4_top50"], qr[r["qid"]]) for r in rows) / len(rows); a4r = sw["benchmark_A4R_depth50"]["ndcg10"]
fig, ax = plt.subplots(figsize=(8, 4.2)); fig.subplots_adjust(left=0.1, right=0.74, top=0.76, bottom=0.2)
ax.axhline(a4r, color=EMB, lw=2); ax.axhline(a4, color=EMB, lw=2, ls="--")
ax.text(640, a4r, f"Embeddings + reranker  {a4r:.3f}", va="center", fontsize=8.5, color=INK2, clip_on=False)
ax.text(640, a4, f"Embeddings  {a4:.3f}", va="center", fontsize=8.5, color=INK2)
ax.plot(D, kw, color=KW, lw=2, marker="o", ms=6, mec=BG)
for d, v in zip(D, kw): ax.text(d, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.text(640, kw[-1], "Keyword + reranker", va="center", fontsize=8.5, color=INK2)
ax.set_xscale("log"); ax.set_xticks(D)
ax.set_xticklabels([f"top {d}\n${sw['depths'][str(d)]['cost_per_search_usd']:.4f}" for d in D])
ax.set_xlim(42, 600); ax.set_ylim(0.34, 0.5); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("How many keyword candidates the reranker reads, and its cost per search"); ax.set_ylabel("nDCG@10")
title(fig, "Reading deeper buys plain embeddings, not reranked ones",
      f"150 FiQA dev questions. Embeddings + reranker reads only its top 50, at ${sw['benchmark_A4R_depth50']['cost_per_search_usd']:.4f} per search.\nGoing from 200 to 500 candidates adds one point for 2.4 times the cost.")
save(fig, "fig6-depth")
print(sorted(p.name for p in OUT.iterdir()))
