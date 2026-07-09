#!/usr/bin/env python3
"""
Full, unaggregated TDColER-benchmark breakdown: every (dataset, classifier)
cell, individually, with no filtering or exclusion.

Produces two tables from results_tdbench_valcheckpoint/results.csv
(p=8, val_checkpoint selector, paper's 5-classifier protocol):

  1. tab:tdbench_full      -- longtable, one row per (dataset, classifier):
                               distilled accuracy, random, full, RR.
  2. tab:tdbench_rr_matrix -- compact table, datasets x classifiers, RR only.

RR is mathematically undefined (0/0) for two cells where full-data and
random-baseline accuracy are both exactly 1.0; these are shown as "--".

Output: prints both to stdout.
"""
import pandas as pd

CSV = "results_tdbench_valcheckpoint_p48/results.csv"
CLFS = ["mlp_sci", "lr", "knn", "nb", "xgboost"]  # paper Table 6 protocol
CLF_HDR = {"mlp_sci": "MLP", "lr": "LR", "knn": "KNN", "nb": "NB",
           "xgboost": "XGB"}


def esc(name):
    return name.replace("_", r"\_")


def fmt_rr(v):
    return "--" if pd.isna(v) else f"{v:.3f}"


def main():
    df = pd.read_csv(CSV)
    df = df[(df.selector == "val_checkpoint") & (df.classifier.isin(CLFS))].copy()
    datasets = sorted(df.dataset.unique())

    # ---------------- Table 1: full breakdown, long format ----------------
    print(r"\begin{longtable}{l l cccc}")
    print(r"\caption{Full per-(dataset, classifier) breakdown on the 23 "
          r"TDBench datasets at IPC=10, validation-selected TAME-LnRes.}")
    print(r"\label{tab:tdbench_full}\\")
    print(r"\toprule")
    print(r"Dataset & Classifier & Accuracy & Random & Full & RR \\")
    print(r"\midrule")
    print(r"\endfirsthead")
    print(r"\toprule")
    print(r"Dataset & Classifier & Accuracy & Random & Full & RR \\")
    print(r"\midrule")
    print(r"\endhead")
    print(r"\bottomrule")
    print(r"\endfoot")
    for ds in datasets:
        sub = df[df.dataset == ds].set_index("classifier")
        for i, clf in enumerate(CLFS):
            if clf not in sub.index:
                continue
            row = sub.loc[clf]
            name = esc(ds) if i == 0 else ""
            print(f"{name} & {CLF_HDR[clf]} & "
                  f"{row.acc_distilled_mean:.3f} & {row.acc_random_ipc10:.3f} & "
                  f"{row.acc_full:.3f} & {fmt_rr(row.rr_tdcoler_formula)} \\\\")
    print(r"\end{longtable}")

    # ---------------- Table 2: RR-only matrix ----------------
    print()
    print(r"\begin{table*}[t]")
    print(r"\centering\small")
    print(r"\caption{Relative regret per dataset and classifier, IPC=10, "
          r"validation-selected TAME-LnRes. The Overall column is the pooled "
          r"median over all (dataset, classifier) cells, matching Table~"
          r"\ref{tab:tdcoler-comparison}.}")
    print(r"\label{tab:tdbench_rr_matrix}")
    ncols = len(CLFS)
    print(r"\begin{tabular}{l " + " ".join(["c"] * ncols) + " c}")
    print(r"\toprule")
    print("Dataset & " + " & ".join(CLF_HDR[c] for c in CLFS) + r" & Overall \\")
    print(r"\midrule")
    for ds in datasets:
        sub = df[df.dataset == ds].set_index("classifier")
        cells = [fmt_rr(sub.loc[clf].rr_tdcoler_formula) if clf in sub.index
                 else "--" for clf in CLFS]
        row_overall = sub.rr_tdcoler_formula.median()
        print(f"{esc(ds)} & " + " & ".join(cells) +
              f" & \\textit{{{row_overall:.3f}}} \\\\")
    print(r"\midrule")
    med_row = " & ".join(f"{df[df.classifier==c].rr_tdcoler_formula.median():.3f}"
                         for c in CLFS)
    overall_pooled = df.rr_tdcoler_formula.median()
    print(f"\\textbf{{median}} & {med_row} & \\textbf{{{overall_pooled:.3f}}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
