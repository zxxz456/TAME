#!/usr/bin/env python3
"""
Generate the large-scale evaluation LaTeX table directly from the result
CSVs. No number is ever hand-transcribed: every cell is read from disk and
formatted here, so the table is guaranteed to match the source files and is
regenerable at any time.

Sources:
  Higgs-940k : results_final_higgs/results.csv
  Airline    : results_final_airline2/results.csv   (may not exist yet)

TAME uses the LnRes embedder. Prints mean (3 dp); pass --std for mean+-std.
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBEDDER = "ln_res_l"
CLFS = ["mlp", "rf", "xgboost"]
IPCS = [10, 50]
# display label -> (method, which accuracy column). The base row keeps the
# paper's established name (TAME-LnRes); the enhancement is marked (best-val).
# "best-loss" is never surfaced as a concept.
ROWS = [
    ("TAME-LnRes",              "tame",           "test_acc_bestloss_mean"),
    ("TAME-LnRes (best-val)",   "tame",           "test_acc_mean"),
    ("Leverage",                "leverage_score", "test_acc_mean"),
    ("VQ",                      "vq",             "test_acc_mean"),
    ("Random",                  "random",         "test_acc_mean"),
    ("Full",                    "full",           "test_acc_mean"),
]
DATASETS = [("Higgs-940k", "results_final_higgs/results.csv"),
            ("Airline Passenger Satisfaction", "results_final_airline2/results.csv")]


def load(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["embedder"] = df["embedder"].fillna("")
    return df


def cell(df, method, col, ipc, clf):
    """Exact value from the CSV, or None if absent."""
    if df is None:
        return None
    emb = EMBEDDER if method == "tame" else ""
    sub = df[(df["method"] == method) & (df["embedder"] == emb)
             & (df["ipc"] == ipc) & (df["classifier"] == clf)]
    if col not in df.columns or sub.empty:
        return None
    v = sub[col].values
    return float(v[0]) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", action="store_true", help="show mean+-std")
    args = ap.parse_args()

    dfs = {name: load(path) for name, path in DATASETS}

    # column order: (dataset, ipc, clf)
    cols = [(dn, ipc, clf) for dn, _ in DATASETS for ipc in IPCS for clf in CLFS]

    # gather raw values per (row_label, col)
    grid = {}
    for label, method, acccol in ROWS:
        for (dn, ipc, clf) in cols:
            grid[(label, dn, ipc, clf)] = cell(dfs[dn], method, acccol, ipc, clf)

    # best distillation method per column (exclude Full), for bolding
    distill_labels = [l for l, _, _ in ROWS if l != "Full"]
    best = {}
    for (dn, ipc, clf) in cols:
        vals = [(l, grid[(l, dn, ipc, clf)]) for l in distill_labels
                if grid[(l, dn, ipc, clf)] is not None]
        if vals:
            best[(dn, ipc, clf)] = max(vals, key=lambda x: x[1])[0]

    def fmt(label, dn, ipc, clf):
        v = grid[(label, dn, ipc, clf)]
        if v is None:
            return "--"
        s = f"{v:.3f}"
        if best.get((dn, ipc, clf)) == label:
            s = f"\\textbf{{{s}}}"
        return s

    # emit
    print(r"\begin{table*}[t]")
    print(r"\centering\small")
    print(r"\begin{tabular}{l ccc ccc ccc ccc}")
    print(r"\toprule")
    print(r"& \multicolumn{6}{c}{\textbf{Higgs-940k}} & "
          r"\multicolumn{6}{c}{\textbf{Airline Passenger Satisfaction}} \\")
    print(r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}")
    print(r"& \multicolumn{3}{c}{IPC=10} & \multicolumn{3}{c}{IPC=50}"
          r" & \multicolumn{3}{c}{IPC=10} & \multicolumn{3}{c}{IPC=50} \\")
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}")
    print(r"Method & MLP & RF & XGB & MLP & RF & XGB & MLP & RF & XGB & MLP & RF & XGB \\")
    print(r"\midrule")
    for label, _, _ in ROWS:
        if label == "Full":
            print(r"\midrule")
        cells = " & ".join(fmt(label, dn, ipc, clf) for (dn, ipc, clf) in cols)
        print(f"{label} & {cells} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")

    # provenance report to stderr (not part of the table)
    print("\n--- provenance ---", file=sys.stderr)
    for name, path in DATASETS:
        status = "MISSING" if dfs[name] is None else f"{len(dfs[name])} rows"
        print(f"{name}: {path} [{status}]", file=sys.stderr)


if __name__ == "__main__":
    main()
