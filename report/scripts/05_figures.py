"""Figure, table and the numbers quoted in the pydoubletfinder technical report.

Usage: python 05_figures.py <runs_dir> <tasks_dir> <out_dir>
Compares DoubletFinder 2.0.6 (Seurat 5) and pydoubletfinder on the Xi & Li (2021) real doublet benchmark,
seeds 1 and 2 (set.seed before paramSweep). Python and R are compared at the same seed; the two R seeds show
how much the result moves with the seed. Panels f-i: UMAP of pbmc-2ctrl-dm (seed 1), for display only.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

RUNS, TASKS, OUT = map(pathlib.Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
C_R, C_PY, C_RR = "#eb6834", "#2a78d6", "#a8a7a1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
ENG = {"R": "DoubletFinder", "py": "pydoubletfinder"}
SEEDS = [1, 2]


def load(engine, task, seed):
    return pd.read_csv(RUNS / ENG[engine] / task / f"seed{seed}" / "scores.tsv.gz", sep="\t")


def meta(engine, task, seed):
    return json.load(open(RUNS / ENG[engine] / task / f"seed{seed}" / "timing.json"))


def jaccard(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


tasks = sorted(p.name for p in TASKS.iterdir() if (p / "labels.tsv").exists())
rows, pks, pool = [], [], []
for t in tasks:
    y = (pd.read_csv(TASKS / t / "labels.tsv", header=None)[0] == "doublet").to_numpy()
    sc = {(e, s): load(e, t, s) for e in ENG for s in SEEDS}
    score = {k: d["score"].to_numpy() for k, d in sc.items()}
    call = {k: (d["class"] == "doublet").to_numpy() for k, d in sc.items()}
    pool.append(np.column_stack([score["R", 1], score["py", 1], score["R", 2]]))
    row = dict(dataset=t, cells=len(y), doublet_rate=y.mean())
    same = [np.abs(score["py", s] - score["R", s]) < 1e-12 for s in SEEDS]
    row["same_pann_pR"] = np.mean([m.mean() for m in same])
    row["maxdiff_pR"] = max(np.abs(score["py", s] - score["R", s]).max() for s in SEEDS)
    row["same_call_pR"] = np.mean([(call["py", s] == call["R", s]).mean() for s in SEEDS])
    row["jaccard_pR"] = np.mean([jaccard(call["py", s], call["R", s]) for s in SEEDS])
    row["same_pann_RR"] = (np.abs(score["R", 1] - score["R", 2]) < 1e-12).mean()
    row["jaccard_RR"] = jaccard(call["R", 1], call["R", 2])
    row["spearman_RR"] = spearmanr(score["R", 1], score["R", 2])[0]
    for e in ENG:
        row[f"auprc_{e}"] = np.mean([average_precision_score(y, score[e, s]) for s in SEEDS])
        row[f"auroc_{e}"] = np.mean([roc_auc_score(y, score[e, s]) for s in SEEDS])
        row[f"sec_{e}"] = min(meta(e, t, s)["seconds"] for s in SEEDS)       # faster of the two runs
        for s in SEEDS:
            pks.append(dict(dataset=t, engine=e, seed=s, pK=meta(e, t, s)["pK"]))
    rows.append(row)
tab = pd.DataFrame(rows).sort_values("cells").reset_index(drop=True)
pk = pd.DataFrame(pks).pivot_table(index=["dataset", "seed"], columns="engine", values="pK").reset_index()
pkR = pk.pivot(index="dataset", columns="seed", values="R")
tab = tab.merge(pkR.rename(columns={1: "pK_R1", 2: "pK_R2"}), left_on="dataset", right_index=True)
tab.to_csv(OUT / "comparison_per_dataset.tsv", sep="\t", index=False)

# ======================= UMAP of one dataset =======================
UMAP_NAME = "pbmc-2ctrl-dm"
td = TASKS / UMAP_NAME
ad = scanpy.AnnData(sp.csr_matrix(scipy.io.mmread(td / "counts.mtx.gz").T))
ad.obs["annotated"] = pd.read_csv(td / "labels.tsv", header=None)[0].to_numpy()
for e in ENG:
    d = load(e, UMAP_NAME, 1)
    ad.obs[f"score_{e}"] = d["score"].to_numpy()
    ad.obs[f"call_{e}"] = (d["class"] == "doublet").to_numpy()
scanpy.pp.normalize_total(ad, target_sum=1e4)
scanpy.pp.log1p(ad)
scanpy.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
scanpy.pp.pca(ad, n_comps=30, random_state=0)
scanpy.pp.neighbors(ad, n_neighbors=15, random_state=0)
scanpy.tl.umap(ad, random_state=0)
xy = ad.obsm["X_umap"]

# ======================= Figure 1 =======================
fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.9, 0.95], hspace=0.5)


def panel(ax, s, dx=-0.36):
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def identity(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.6, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def crop(img):
    """Trim transparent/white margins so the logo fills its axes."""
    ink = (img[..., 3] > 0.04) & (img[..., :3].min(-1) < 0.96)
    ys, xs = np.where(ink)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


top = gs[0].subgridspec(1, 4, width_ratios=[0.62, 1, 1, 1], wspace=0.36)
logos = [crop(plt.imread(ASSETS / f)) for f in ["pydoubletfinder_logo.png", "omicverse_logo.png"]]

axa = fig.add_subplot(top[1])
identity(axa, 0, 1)
axa.scatter(tab.auprc_R, tab.auprc_py, s=12, color=C_PY, edgecolor="white", linewidth=0.3)
axa.set_title("AUPRC")
axa.set_xlabel("DoubletFinder (R)")
axa.set_ylabel("pydoubletfinder")
panel(axa, "a")

axb = fig.add_subplot(top[2])
identity(axb, 0, 0.32)
jit = np.random.default_rng(0).uniform(-0.004, 0.004, len(tab))
axb.scatter(tab.pK_R1 + jit, tab.pK_R2 - jit, s=12, color=C_RR, edgecolor="white", linewidth=0.3,
            label="R seed 1 vs R seed 2", zorder=2)
axb.scatter(pk.R, pk.py, s=12, color=C_PY, edgecolor="white", linewidth=0.3, label="Python vs R, same seed", zorder=3)
axb.set_title("Chosen pK (BCmvn)")
axb.set_xlabel("R")
axb.set_ylabel("Python, or R seed 2")
axb.legend(fontsize=5, loc="lower right", handletextpad=0.1, borderaxespad=0.1)
panel(axb, "b")

axc = fig.add_subplot(top[3])
pool = np.vstack(pool)
identity(axc, 0, 1)
axc.scatter(pool[:, 0], pool[:, 2], s=0.3, color=C_RR, lw=0, rasterized=True, label="R seed 2")
axc.scatter(pool[:, 0], pool[:, 1], s=0.3, color=C_PY, lw=0, rasterized=True, label="Python, same seed")
axc.set_title(f"pANN of all {len(pool):,} cells")
axc.set_xlabel("R seed 1")
axc.set_ylabel("Python, or R seed 2")
leg = axc.legend(fontsize=5, loc="lower right", handletextpad=0.1, borderaxespad=0.1, markerscale=6)
panel(axc, "c")

bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.3)
axd = fig.add_subplot(bot[0])
axd.scatter(tab.cells, tab.sec_R, s=10, color=C_R, edgecolor="white", linewidth=0.3, label="DoubletFinder 2.0.6 (R)")
axd.scatter(tab.cells, tab.sec_py, s=10, color=C_PY, edgecolor="white", linewidth=0.3, label="pydoubletfinder")
axd.set_xscale("log")
axd.set_yscale("log")
axd.set_xlabel("cells")
axd.set_yticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
axd.minorticks_off()
axd.set_ylabel("wall time, 1 thread (s)")
axd.set_title("Run time, paramSweep to doubletFinder")
axd.grid(color=GRID, lw=0.5)
axd.set_axisbelow(True)
axd.legend(fontsize=5.5, loc="upper left", handletextpad=0.1)
panel(axd, "d", dx=-0.2)

axe = fig.add_subplot(bot[1])
axe.scatter(tab.cells, tab.jaccard_RR, s=12, color=C_RR, edgecolor="white", linewidth=0.3, label="R seed 1 vs R seed 2")
axe.scatter(tab.cells, tab.jaccard_pR, s=12, color=C_PY, edgecolor="white", linewidth=0.3, label="Python vs R, same seed")
axe.set_xscale("log")
axe.set_ylim(0, 1.05)
axe.set_xlabel("cells")
axe.set_ylabel("Jaccard index")
axe.set_title("Overlap of doublet calls")
axe.grid(color=GRID, lw=0.5)
axe.set_axisbelow(True)
axe.legend(fontsize=5.5, loc="lower left", handletextpad=0.1)
panel(axe, "e", dx=-0.2)

row3 = gs[2].subgridspec(1, 4, wspace=0.06)
SIZE = {"R only": 2.2, "Python only": 2.2}


def umap_ax(k, title, letter):
    ax = fig.add_subplot(row3[k])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(title)
    ax.set_aspect("equal")
    panel(ax, letter, dx=-0.04)
    return ax


def cats(ax, labels, order, colors):
    for name, col in zip(order, colors):
        m = labels == name
        ax.scatter(xy[m, 0], xy[m, 1], s=SIZE.get(name, 0.7), c=col, lw=0, rasterized=True,
                   label=f"{name} ({m.sum():,})")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=5.5, handletextpad=0.1,
                    columnspacing=0.8)
    for h in leg.legend_handles:
        h.set_sizes([10])


ann = ad.obs["annotated"].to_numpy()
cats(umap_ax(0, "Annotated doublets", "f"), ann, ["singlet", "doublet"], ["#d9d8d3", "#2a2a2a"])
for k, (e, title, letter) in enumerate([("R", "pANN, R", "g"), ("py", "pANN, Python", "h")]):
    ax = umap_ax(k + 1, title, letter)
    v = ad.obs[f"score_{e}"].to_numpy()
    o = np.argsort(v, kind="stable")
    im = ax.scatter(xy[o, 0], xy[o, 1], c=v[o], s=0.6, cmap="magma_r", vmin=0, vmax=v.max(), lw=0, rasterized=True)
    if k == 0:
        cax = ax.inset_axes([0.55, -0.14, 0.9, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.5, length=1.5)
umap_same = (np.abs(ad.obs.score_R - ad.obs.score_py) < 1e-12).mean()
cax.set_title(f"pANN (identical in {umap_same * 100:.1f}% of cells)", fontsize=5.5, color=INK2, pad=2)
r_, p_ = ad.obs.call_R.to_numpy(), ad.obs.call_py.to_numpy()
ucall = np.select([r_ & p_, r_ & ~p_, ~r_ & p_], ["both", "R only", "Python only"], "neither")
cats(umap_ax(3, "Doublet calls", "i"), ucall, ["neither", "both", "R only", "Python only"],
     ["#d9d8d3", "#2a2a2a", "#eb6834", "#2a78d6"])

# logos: from the left edge of panel letter d to the y label of panel a, from title height to the x labels
pa, pd_ = axa.get_position(), axd.get_position()
x0, x1 = pd_.x0 - 0.2 * pd_.width, pa.x0 - 0.075
y0, y1 = pa.y0 - 0.02, pa.y1 + 0.035
h_ov = (x1 - x0) * 7.2 / 6.2 * logos[1].shape[0] / logos[1].shape[1]
for img, rect, anchor in [(logos[0], [x0, y0 + h_ov + 0.012, x1 - x0, y1 - y0 - h_ov - 0.012], "N"),
                          (logos[1], [x0, y0, x1 - x0, h_ov], "S")]:
    axl = fig.add_axes(rect)
    axl.imshow(img)
    axl.set_anchor(anchor)
    axl.set_axis_off()
fig.savefig(OUT / "fig1.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig1.png", bbox_inches="tight", pad_inches=0.02, dpi=200)

# ======================= numbers =======================
speed = tab.sec_R / tab.sec_py
big = tab.iloc[tab.cells.idxmax()]
n_runs = len(pk)
macros = {
    "nDatasets": len(tab), "nCellsMin": f"{int(tab.cells.min()):,}", "nCellsMax": f"{int(tab.cells.max()):,}",
    "nCellsTotal": f"{int(tab.cells.sum()):,}",
    "rateMin": f"{tab.doublet_rate.min() * 100:.0f}", "rateMax": f"{tab.doublet_rate.max() * 100:.0f}",
    "auprcR": f"{tab.auprc_R.mean():.3f}", "auprcPy": f"{tab.auprc_py.mean():.3f}",
    "aurocR": f"{tab.auroc_R.mean():.3f}", "aurocPy": f"{tab.auroc_py.mean():.3f}",
    "auprcDiffMax": f"{(tab.auprc_py - tab.auprc_R).abs().max():.1e}",
    "nRuns": n_runs // 2, "nSamePK": int((pk.R == pk.py).sum()),
    "nSamePKRR": int((tab.pK_R1 == tab.pK_R2).sum()),
    "samePannMin": f"{tab.same_pann_pR.min() * 100:.1f}", "samePannMinName": tab.dataset[tab.same_pann_pR.idxmin()],
    "samePannAll": f"{(tab.same_pann_pR * tab.cells).sum() / tab.cells.sum() * 100:.2f}",
    "maxPannDiff": f"{tab.maxdiff_pR.max():.3f}",
    "sameCallMin": f"{tab.same_call_pR.min() * 100:.2f}",
    "nSameCallAll": int((tab.same_call_pR == 1).sum()),
    "jPRmin": f"{tab.jaccard_pR.min():.3f}", "jRRmed": f"{tab.jaccard_RR.median():.2f}",
    "jRRmin": f"{tab.jaccard_RR.min():.2f}",
    "speedMin": f"{speed.min():.1f}", "speedMax": f"{speed.max():.1f}", "speedMed": f"{speed.median():.1f}",
    "tBigR": f"{big.sec_R:.0f}", "tBigPy": f"{big.sec_py:.0f}", "bigName": big.dataset,
    "bigCells": f"{int(big.cells):,}",
    "umapName": UMAP_NAME, "umapCells": f"{ad.n_obs:,}", "umapAnnot": f"{(ann == 'doublet').sum():,}",
    "umapSame": f"{umap_same * 100:.1f}", "umapBoth": f"{(ucall == 'both').sum():,}",
    "umapROnly": f"{(ucall == 'R only').sum():,}", "umapPyOnly": f"{(ucall == 'Python only').sum():,}",
}
with open(OUT / "numbers.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(OUT / "numbers.tex").read())
print(tab[["dataset", "cells", "doublet_rate", "auprc_R", "auprc_py", "pK_R1", "pK_R2", "same_pann_pR", "maxdiff_pR",
           "same_call_pR", "jaccard_RR", "sec_R", "sec_py"]].round(4).to_string())
