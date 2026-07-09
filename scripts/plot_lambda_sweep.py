#!/usr/bin/env python3
"""
Figure: downstream accuracy vs the covariance weight lambda (Eq. 7), per
classifier, mean over the 5 sweep datasets and both IPC budgets. Categorical
x-axis so lambda=0 (mean-only, the Distribution-Matching objective) is shown
alongside the log-spaced positive values. The default lambda=1 is marked.

Input:  results_lambda_sweep/results.csv
Output: results_lambda_sweep/fig_lambda_sweep.{png,pdf}
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results_lambda_sweep"
df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))

LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0]
XLAB = ["0", "0.1", "0.5", "1", "2", "10"]
x = np.arange(len(LAMBDAS))
CLF = {"mlp": ("MLP", "#1f77b4"), "rf": ("Random Forest", "#2ca02c"),
       "xgboost": ("XGBoost", "#d62728")}

fig, ax = plt.subplots(figsize=(5.4, 3.8))
for clf, (label, color) in CLF.items():
    sub = df[df.classifier == clf]
    means = [sub[sub.cov_weight == lam].test_acc_mean.mean() for lam in LAMBDAS]
    ax.plot(x, means, marker="o", color=color, lw=2, label=label)

ax.axvline(3, color="gray", ls=":", lw=1.2)  # default lambda=1 at index 3
ax.text(3.02, ax.get_ylim()[0], r" default $\lambda=1$", color="gray",
        fontsize=8, va="bottom")
ax.set_xticks(x); ax.set_xticklabels(XLAB)
ax.set_xlabel(r"covariance weight $\lambda$")
ax.set_ylabel("Mean downstream accuracy")
ax.set_title(r"Sensitivity to the covariance weight $\lambda$", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="lower center")
fig.tight_layout()

for ext in ["png", "pdf"]:
    out = os.path.join(RESULTS_DIR, f"fig_lambda_sweep.{ext}")
    fig.savefig(out, dpi=200)
    print("saved:", out)
