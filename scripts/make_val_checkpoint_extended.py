#!/usr/bin/env python3
"""
Extended per-dataset tables for validation-based snapshot selection.
One table per IPC: all 18 datasets x (base TAME-LnRes vs best-val) for each
of MLP/RF/XGBoost, with mean and median summary rows. Backs the aggregate
Table tab:val_checkpoint. Every number read from results_val_checkpoint.

Cell = mean test accuracy over the 3 runs for that dataset. Summary rows =
mean / median over the 18 per-dataset values.
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = "results_val_checkpoint/results.csv"
CLFS = ["mlp", "rf", "xgboost"]
CLF_HDR = {"mlp": "MLP", "rf": "RF", "xgboost": "XGB"}


def main():
    df = pd.read_csv(CSV)
    # per (dataset, ipc, classifier): mean over runs of base and best-val
    g = (df.groupby(["dataset", "ipc", "classifier"])[["best_loss", "best_val"]]
         .mean().reset_index())

    for ipc in [10, 50]:
        sub = g[g["ipc"] == ipc]
        datasets = sorted(sub["dataset"].unique())
        # wide: dataset -> {clf: (base, val)}
        def val(ds, clf, col):
            r = sub[(sub.dataset == ds) & (sub.classifier == clf)]
            return float(r[col].iloc[0]) if len(r) else float("nan")

        print(f"% ===== IPC={ipc} =====")
        print(r"\begin{table*}[t]")
        print(r"\centering\small")
        print(rf"\caption{{Per-dataset validation-based snapshot selection at "
              rf"IPC={ipc}: base TAME-LnRes vs.\ best-val, mean test accuracy "
              rf"over 3 runs. Summary rows are mean / median over the 18 "
              rf"datasets.}}")
        print(rf"\label{{tab:valext_ipc{ipc}}}")
        print(r"\begin{tabular}{l cc cc cc}")
        print(r"\toprule")
        print(r"& \multicolumn{2}{c}{MLP} & \multicolumn{2}{c}{RF}"
              r" & \multicolumn{2}{c}{XGBoost} \\")
        print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
        print(r"Dataset & base & best-val & base & best-val & base & best-val \\")
        print(r"\midrule")
        for ds in datasets:
            cells = []
            for clf in CLFS:
                b, v = val(ds, clf, "best_loss"), val(ds, clf, "best_val")
                vb = f"\\textbf{{{v:.3f}}}" if v > b else f"{v:.3f}"
                cells += [f"{b:.3f}", vb]
            print(f"{ds} & " + " & ".join(cells) + r" \\")
        print(r"\midrule")
        # mean and median over the 18 per-dataset values
        for stat, fn in [("mean", "mean"), ("median", "median")]:
            cells = []
            for clf in CLFS:
                cc = sub[sub.classifier == clf]
                b = getattr(cc["best_loss"], fn)()
                v = getattr(cc["best_val"], fn)()
                cells += [f"{b:.3f}", f"\\textbf{{{v:.3f}}}"]
            print(f"\\textit{{{stat}}} & " + " & ".join(cells) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table*}")
        print()

    # reconciliation: aggregate median over 54 rows (as in tab:val_checkpoint)
    print("--- reconciliation (stderr) ---", file=sys.stderr)
    for ipc in [10, 50]:
        for clf in CLFS:
            s = df[(df.ipc == ipc) & (df.classifier == clf)]
            sub = g[(g.ipc == ipc) & (g.classifier == clf)]
            print(f"IPC{ipc} {clf}: best_val median over 54 rows="
                  f"{s.best_val.median():.3f}  over 18 ds-means="
                  f"{sub.best_val.median():.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
