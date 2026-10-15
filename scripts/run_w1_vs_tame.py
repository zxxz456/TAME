"""Plain TAME against TAME with the W1 marginal term, on shared seeds.

The question is whether matching each column's marginal in data space lifts the
tree classifiers on datasets with no categorical columns, where the one-hot
fix has nothing to act on. On those, plain TAME beats random real rows by 1 to
5 points on trees in binary tasks, and in multi-class it beats them only at
IPC 10: at IPC 50 and 200 it falls below them, by up to 2 points with
xgboost (the null blocks of `run_onehot_modes.py`). The paper puts the tree gap
down to Gaussianisation of the marginals (section 5.3), which grows with the
optimisation; this term pulls the marginals back towards the real ones every
iteration.

Two arms by default, `none` and `w1@<weight>`, with the weight from
`run_w1_tuning.py` (validation, seeds 1000-1009). It is required on the command
line on purpose, so this cannot run with an untuned value by accident. Same
grid as the one-hot comparison: IPC 10, 50 and 200, ln_res_l and dcnv2_base,
ten seeds (132-141), 1000 iterations, mlp, rf and xgboost, every set read as
the lowest-loss set, the last iterate and the initialisation.

Each row also stores the marginal measures of w1_common.py, w1 and Shapiro-Wilk,
for the three sets. They separate "the term did not help" from "the term did
not do what it is for": if accuracy does not move, the w1 and sw columns say
whether the marginals moved at all.

`--arms` takes any list of arm labels, which is how the ste x W1 design on the
mixed datasets will reuse this script.

    THREADS=3 POST=analyze_w1.py bash scripts/run_sharded.sh run_w1_vs_tame.py \\
      ~/tame_runs/w1_vs_tame 5 magic spambase credit madelon higgs \\
      letter pendigits satimage segment pageblocks -- --weight <tuned>
"""
import os, sys, argparse
import pandas as pd
sys.path.insert(0, "/home/zxxz6/TAME"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/zxxz6/TAME")
from w1_common import BIN_SIN, MULTI_SIN, correr, linea, reanudar

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=BIN_SIN + MULTI_SIN)
ap.add_argument("--weight", type=float, required=True,
                help="the W1 weight chosen by run_w1_tuning.py")
ap.add_argument("--arms", nargs="+", default=None,
                help="defaults to none and w1@<weight>")
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--embedders", nargs="+", default=["ln_res_l", "dcnv2_base"])
ap.add_argument("--normalidad", action="store_true",
                help="also measure the paper's per-class Shapiro-Wilk and Mardia on every set")
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/w1_vs_tame"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
BRAZOS = a.arms or ["none", f"w1@{a.weight:g}"]

csv = os.path.join(a.out, "results.csv")
rows, hecho = reanudar(csv, ["dataset", "embedder", "ipc", "brazo", "seed"])
for ds in a.datasets:
    print(f"##### {ds} | {BRAZOS}", flush=True)
    for emb in a.embedders:
        for ipc in a.ipcs:
            for etq in BRAZOS:
                for run in range(a.runs):
                    seed = 132 + run
                    if (ds, emb, ipc, etq, seed) in hecho:
                        continue
                    nuevas = correr(ds, emb, ipc, etq, seed, a.iters, medir_normalidad=a.normalidad)
                    for r in nuevas:
                        r["run"] = run
                    rows += nuevas
                    print(linea(nuevas), flush=True)
                    pd.DataFrame(rows).to_csv(csv, index=False)

df = pd.DataFrame(rows)
df.to_csv(csv, index=False)
pd.set_option("display.width", 240)
print("\n=== test accuracy, lowest-loss set, mean per arm ===")
print(df.pivot_table(index=["bloque", "classifier"], columns="brazo",
                     values="acc_bestloss").round(4).to_string())
print("\n=== marginals: W1 to the real columns and Shapiro-Wilk, init -> returned set ===")
print(df.drop_duplicates(["dataset", "embedder", "ipc", "brazo", "seed"])
      .groupby(["bloque", "brazo"])[["w1_init", "w1_bestloss", "sw_init", "sw_bestloss"]]
      .mean().round(4).to_string())
print(f"\nCSV -> {csv}")
