#!/usr/bin/env python3
"""
Per-dataset TDColER-benchmark table, val-checkpoint selector only.
One row per dataset: distilled accuracy, random@IPC=10 accuracy, full-data
accuracy, and relative regret -- each averaged over the paper's 5-classifier
protocol (MLP, LR, KNN, NB, XGBoost; RF excluded, matching Table 6).

Rows with a non-positive RR denominator (acc_full <= acc_random) are marked
with a dagger rather than dropped, so the full 23-dataset picture is visible.

Input:  results_tdbench_valcheckpoint/results.csv  (p=8, canonical protocol)
Output: prints the LaTeX table to stdout
"""
import pandas as pd

CSV = "results_tdbench_valcheckpoint_p48/results.csv"
CLFS = ["mlp_sci", "lr", "knn", "nb", "xgboost"]  # paper Table 6 protocol


def main():
    df = pd.read_csv(CSV)
    df = df[(df.selector == "val_checkpoint") & (df.classifier.isin(CLFS))]

    g = df.groupby("dataset").agg(
        acc=("acc_distilled_mean", "mean"),
        random=("acc_random_ipc10", "mean"),
        full=("acc_full", "mean"),
        rr=("rr_tdcoler_formula", "mean"),
        denom_min=("acc_full", lambda s: None),  # placeholder, computed below
    ).drop(columns="denom_min")

    # per-dataset flag: any classifier with non-positive denominator
    df["denom"] = df.acc_full - df.acc_random_ipc10
    flagged = df.groupby("dataset")["denom"].apply(lambda s: (s <= 0).any())
    g["flag"] = flagged

    g = g.sort_index()

    print(r"\begin{table}[t]")
    print(r"\centering\small")
    print(r"\caption{Per-dataset results on the 23 TDBench datasets at IPC=10, "
          r"validation-selected TAME-LnRes, averaged over the five-classifier "
          r"protocol (MLP, LR, KNN, NB, XGBoost). \textdagger{} marks datasets "
          r"where at least one classifier has a non-positive relative-regret "
          r"denominator (Section~<<sec:rr_caveat>>). ``median of per-dataset "
          r"rows'' summarizes the per-dataset rows above (each already "
          r"averaged over classifiers). ``pooled median (all cells)'' and "
          r"``pooled median (valid cells)'' are computed over the individual "
          r"(dataset, classifier) cells directly, matching the Overall column "
          r"of Table~\ref{tab:tdcoler}.}")
    print(r"\label{tab:tdbench_perdataset}")
    print(r"\begin{tabular}{l cccc}")
    print(r"\toprule")
    print(r"Dataset & Accuracy & Random & Full & RR \\")
    print(r"\midrule")
    for ds, row in g.iterrows():
        mark = r"\textdagger{}" if row["flag"] else ""
        name = ds.replace("_", r"\_") + mark
        print(f"{name} & {row['acc']:.3f} & {row['random']:.3f} & "
              f"{row['full']:.3f} & {row['rr']:.3f} \\\\")
    # filtered median: over individual valid CELLS (denom>0), matching the
    # cell-level filtering used for the aggregate Table 6 numbers -- not a
    # median of the per-dataset rows above, which stay unfiltered for
    # visibility.
    valid = df[df.denom > 0]

    print(r"\midrule")
    print(f"\\textit{{median of per-dataset rows}} & {g['acc'].median():.3f} & "
          f"{g['random'].median():.3f} & {g['full'].median():.3f} & "
          f"{g['rr'].median():.3f} \\\\")
    print(f"\\textit{{pooled median (all cells)}} & "
          f"{df['acc_distilled_mean'].median():.3f} & "
          f"{df['acc_random_ipc10'].median():.3f} & "
          f"{df['acc_full'].median():.3f} & "
          f"{df['rr_tdcoler_formula'].median():.3f} \\\\")
    print(f"\\textit{{pooled median (valid cells)}} & "
          f"{valid['acc_distilled_mean'].median():.3f} & "
          f"{valid['acc_random_ipc10'].median():.3f} & "
          f"{valid['acc_full'].median():.3f} & "
          f"{valid['rr_tdcoler_formula'].median():.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
