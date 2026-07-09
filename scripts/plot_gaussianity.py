#!/usr/bin/env python3
"""
Mechanism figure: Shapiro-Wilk pass rate of the distilled set vs distillation
iteration (median + IQR across datasets), with two annotated exemplar
datasets: adult (categorical, tree accuracy declines after ~100 iters) and
pendigits (continuous, accuracy rises monotonically).

Input:  results_gaussianity_traj/results.csv
Output: results_gaussianity_traj/fig_gaussianity_convergence.{png,pdf}
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results_gaussianity_traj"
df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))
df = df[df["iter"] >= -1]

# average runs -> one trajectory per (dataset, ipc, iter)
per = (df.groupby(["ipc", "dataset", "iter"])["shapiro_pass_rate"]
       .mean().reset_index())

EXEMPLARS = {}  # median + IQR over all 18 datasets only; no per-dataset lines

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
for ax, ipc in zip(axes, [10, 50]):
    sub = per[per["ipc"] == ipc]
    med = sub.groupby("iter")["shapiro_pass_rate"].median()
    q25 = sub.groupby("iter")["shapiro_pass_rate"].quantile(0.25)
    q75 = sub.groupby("iter")["shapiro_pass_rate"].quantile(0.75)
    ax.plot(med.index, med.values, color="#1f77b4", lw=2,
            label="median over 18 datasets")
    ax.fill_between(med.index, q25.values, q75.values,
                    color="#1f77b4", alpha=0.15)
    for ds, (color, label) in EXEMPLARS.items():
        g = sub[sub["dataset"] == ds].set_index("iter")["shapiro_pass_rate"]
        ax.plot(g.index, g.values, color=color, lw=1.4, ls="--", label=label)
    ax.set_title(f"IPC = {ipc}")
    ax.set_xlabel("Distillation iteration")
    ax.set_xlim(-1, 1000)

axes[0].set_ylabel("Shapiro-Wilk pass rate (higher = more Gaussian)")
axes[0].legend(loc="lower right", frameon=False, fontsize=8)
fig.suptitle("Distilled sets become more Gaussian as moment matching "
             "proceeds (data-space normality per iteration)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])

for ext in ["png", "pdf"]:
    out = os.path.join(RESULTS_DIR, f"fig_gaussianity_convergence.{ext}")
    fig.savefig(out, dpi=200)
    print("saved:", out)
