#!/usr/bin/env python3
"""
Figure: test-accuracy gain over the initialization vs distillation iteration,
per downstream classifier, with an inter-quartile band across datasets.
Vertical ticks mark the median validation-selected iteration per classifier.

Input:  results_val_checkpoint/snapshots.csv, results.csv
Output: results_val_checkpoint/fig_val_checkpoint_convergence.{png,pdf}
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results_val_checkpoint"

snaps = pd.read_csv(os.path.join(RESULTS_DIR, "snapshots.csv"))
res = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))

# gain over init, per (dataset, run, classifier, iter)
opt = snaps[snaps["iter"] >= 0].copy()
init = snaps[snaps["iter"] == -1][
    ["dataset", "ipc", "run", "classifier", "test_acc"]
].rename(columns={"test_acc": "init_acc"})
opt = opt.merge(init, on=["dataset", "ipc", "run", "classifier"])
opt["delta"] = opt["test_acc"] - opt["init_acc"]

# average runs first -> one trajectory per (dataset, classifier, iter)
per_ds = (opt.groupby(["ipc", "classifier", "dataset", "iter"])["delta"]
          .mean().reset_index())

CLF_LABELS = {"mlp": "MLP", "rf": "Random Forest", "xgboost": "XGBoost"}
COLORS = {"mlp": "#1f77b4", "rf": "#2ca02c", "xgboost": "#d62728"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

for ax, ipc in zip(axes, [10, 50]):
    sub = per_ds[per_ds["ipc"] == ipc]
    for clf in ["mlp", "rf", "xgboost"]:
        g = sub[sub["classifier"] == clf]
        med = g.groupby("iter")["delta"].median()
        q25 = g.groupby("iter")["delta"].quantile(0.25)
        q75 = g.groupby("iter")["delta"].quantile(0.75)
        iters = med.index.values
        ax.plot(iters, med.values, color=COLORS[clf],
                label=CLF_LABELS[clf], lw=2)
        ax.fill_between(iters, q25.values, q75.values,
                        color=COLORS[clf], alpha=0.12)
        # validation-selected checkpoint: median selected iteration and the
        # median per-dataset gain the selection actually achieves (rides each
        # dataset's own peak, hence above the fixed-iteration median curve)
        sel = res[(res["ipc"] == ipc) & (res["classifier"] == clf)]
        sel_ds = sel.groupby("dataset")[["init", "best_val", "best_val_iter"]].mean()
        sel_gain = (sel_ds["best_val"] - sel_ds["init"]).median()
        med_it = sel["best_val_iter"].median()
        ax.plot([med_it], [sel_gain], marker="*", ms=16, color=COLORS[clf],
                mec="black", mew=0.8, zorder=5)

    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_title(f"IPC = {ipc}")
    ax.set_xlabel("Distillation iteration")
    ax.set_xlim(0, 1000)

axes[0].set_ylabel("Test accuracy gain over initialization")
from matplotlib.lines import Line2D
handles, labels = axes[0].get_legend_handles_labels()
handles.append(Line2D([], [], marker="*", ms=13, color="gray", mec="black",
                      lw=0, label="validation-selected"))
axes[0].legend(handles=handles, loc="lower right", frameon=False)
n_datasets = per_ds["dataset"].nunique()
fig.suptitle(f"Convergence of distilled-set quality across {n_datasets} datasets "
             "(median line, inter-quartile band; stars: median gain of the "
             "validation-selected checkpoint)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])

for ext in ["png", "pdf"]:
    out = os.path.join(RESULTS_DIR, f"fig_val_checkpoint_convergence.{ext}")
    fig.savefig(out, dpi=200)
    print("saved:", out)
