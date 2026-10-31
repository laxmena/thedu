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

# Figure 1: the pipeline
fig, ax = plt.subplots(figsize=(9, 2.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
steps = [("Question", "", INK2), ("Rewrite", "optional, LLM", KW), ("Find top 100", "keyword index\nor embeddings", INK2),
         ("Reorder top 50", "optional, reranker", INK2), ("Top 10", "", INK2)]
x = 2
for i, (h, sub, c) in enumerate(steps):
    w = 16.5
    ax.add_patch(FancyBboxPatch((x, 9), w, 12, boxstyle="round,pad=0,rounding_size=1.5", fc=BG, ec=c, lw=1.4,
                                ls="--" if "optional" in sub else "-"))
    ax.text(x + w / 2, 17.3 if sub else 15, h, ha="center", va="center", fontsize=9, color=INK, fontweight="bold")
    if sub: ax.text(x + w / 2, 12.4, sub, ha="center", va="center", fontsize=8, color=INK2)
    if i < len(steps) - 1: ax.annotate("", xy=(x + w + 3.2, 15), xytext=(x + w + 0.4, 15),
                                       arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    x += w + 3.6
ax.text(2, 3, "Dashed boxes are optional. Under the strict rule no model touches the documents,\nso the only thing built ahead of time is the keyword index.", fontsize=8, color=INK2, va="center")
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
        ax.text(p["hi"] + 0.012, y, f"{lab}  {a:.2f} → {p['mean']:.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(DS))); ax.set_yticklabels([n for _, n in DS][::-1], color=INK)
ax.set_xlim(0.2, 1.2); ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("nDCG@10   (hollow: as ranked; filled: top 100 perfectly reordered, with 95% range)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
title(fig, "Keyword search finds the right documents but ranks them low",
      "Reordered perfectly, keyword search's top 100 would beat embeddings as they rank today.\nEmbeddings' own top 100 could do better still: the first sign the gap is in the candidates.")
save(fig, "fig3-ceilings")

# Figure 4: the fair fight, paired difference vs embeddings + reranker
fig, ax = plt.subplots(figsize=(8, 4.0)); fig.subplots_adjust(left=0.14, right=0.96, top=0.76, bottom=0.15)
ax.axvspan(-0.01, 0, color=GRID, alpha=0.7, lw=0); ax.axvline(0, color=INK2, lw=1)
ax.text(-0.005, -0.42, "within 1 point", ha="center", va="bottom", fontsize=7.5, color=INK2)
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
      "About a point behind on SciFact and NFCorpus, eight to nine on FiQA. Dots are paired means\nover every dev question; lines are 95% ranges.")
save(fig, "fig4-fair-fight")

# Figure 5: made-up names
m = j("m5-results.json")["ndcg10"]; lv = [0, 25, 50, 100]
fig, axes = plt.subplots(1, 2, figsize=(8, 4.0), sharey=True); fig.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.15, wspace=0.08)
series = [("A0R", KW, "-", "Keyword + reranker"), ("A4", EMB, "--", "Embeddings"), ("A4R", EMB, "-", "Embeddings + reranker")]
for ax, cond, sub in [(axes[0], "A", "Users type the codename"), (axes[1], "B", "Users describe it instead")]:
    for arm, c, ls, lab in series:
        ys = [m[f"L{l}{cond}"][arm] for l in lv]
        ax.plot(lv, ys, color=c, ls=ls, lw=2, marker="o", ms=5, mec=BG, label=lab)
    ax.set_title(sub, loc="left", fontsize=10, color=INK)
    ax.set_xticks(lv); ax.set_xticklabels([f"{l}%" for l in lv]); ax.set_xlim(-5, 105)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlabel("Names swapped for codenames")
axes[0].set_ylabel("nDCG@10"); axes[0].set_ylim(0.1, 0.6)
axes[0].legend(loc="lower center", bbox_to_anchor=(1.04, 1.09), ncol=3, frameon=False, fontsize=8.5, handlelength=3)
title(fig, "Codenames hurt embeddings, but don't flip the result",
      "94 FiQA questions that mention one of 54 swapped names. When users type the codename, the gap\nnarrows from 0.075 to 0.013. When they describe it, keyword search has nothing to match.")
save(fig, "fig5-codenames")
print(sorted(p.name for p in OUT.iterdir()))
