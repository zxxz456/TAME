#!/usr/bin/env python3
"""
Bar chart companion to Table tab:val_checkpoint: paired bars (base TAME-LnRes
vs. best-val) per classifier, one panel per IPC. Same numbers as the table,
visual form. Matches the two-panel layout of fig_val_checkpoint_convergence.

Input:  results_val_checkpoint/results.csv
Output: results_val_checkpoint/fig_val_checkpoint_bars.{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results_val_checkpoint"
df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))

CLFS = ["mlp", "rf", "xgboost"]
CLF_LABELS = {"mlp": "MLP", "rf": "RF", "xgboost": "XGBoost"}
COLOR_BASE = "#9ecae1"
COLOR_VAL = "#08519c"

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)

for ax, ipc in zip(axes, [10, 50]):
    sub = df[df.ipc == ipc]
    base_means = [sub[sub.classifier == c].best_loss.mean() for c in CLFS]
    val_means = [sub[sub.classifier == c].best_val.mean() for c in CLFS]

    x = np.arange(len(CLFS))
    w = 0.35
    b1 = ax.bar(x - w/2, base_means, w, label="TAME-LnRes", color=COLOR_BASE)
    b2 = ax.bar(x + w/2, val_means, w, label="best-val", color=COLOR_VAL)

    for xi, (bm, vm) in enumerate(zip(base_means, val_means)):
        gain = vm - bm
        ax.annotate(f"+{gain:.3f}", xy=(xi + w/2, vm), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=COLOR_VAL, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([CLF_LABELS[c] for c in CLFS])
    ax.set_title(f"IPC = {ipc}")
    ax.set_ylim(0.65, 0.87)

axes[0].set_ylabel("Mean test accuracy")
axes[0].legend(frameon=False, loc="upper left", fontsize=9)
fig.suptitle("Validation-based snapshot selection: base vs. best-val "
             "(18 benchmark datasets)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])

for ext in ["png", "pdf"]:
    out = os.path.join(RESULTS_DIR, f"fig_val_checkpoint_bars.{ext}")
    fig.savefig(out, dpi=200)
    print("saved:", out)
