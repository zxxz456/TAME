"""Plain TAME against TAME with the in-loop snap to observed values, on shared seeds.

The multi-class diagnosis traced most of TAME's loss against random real rows at
high IPC to broken ties (hypothesis E): one SGD step makes every value distinct,
and in columns with few values a tree then splits real ties on noise. Snapping
the returned set back to observed values closed letter's gap exactly but cost
credit 4 points, which would force a per-dataset choice. This tests the
alternative that needs none: the snap inside the loop, always on
(`dm_snap`, synth/snap.py), built like the one-hot ste.

Every set is also measured with the paper's per-class Shapiro-Wilk and Mardia
(w1_common.normalidad), besides accuracy and the W1 distance to the real
marginals.

Arms:

  none            plain TAME
  snap            the snap inside the loop
  none+redondeo   plain TAME's sets snapped after distillation, always
  none+valid      the same sets, snapped or not, whichever scores higher on
                  validation, per set and classifier

The last two come from the `none` distillations and cost only classifier fits;
they are there to compare against, not as candidates.

The snap has no hyperparameter, so there is no tuning phase: straight to test,
seeds 132-141. Same grid as the W1 comparison: the 10 datasets with no
categorical columns, IPC 10/50/200, ln_res_l and dcnv2_base. On magic and higgs
the columns are continuous and the grid is so dense that the snap barely moves
anything, so there it should come out at zero.

    POST=analyze_w1.py bash scripts/run_sharded.sh run_snap_vs_tame.py \\
      ~/tame_runs/snap_vs_tame 7 letter higgs pendigits satimage segment credit \\
      magic spambase madelon pageblocks
"""
import os, sys, argparse
import pandas as pd
sys.path.insert(0, "/home/zxxz6/TAME"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/zxxz6/TAME")
from w1_common import BIN_SIN, MULTI_SIN, correr, linea, reanudar

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=BIN_SIN + MULTI_SIN)
ap.add_argument("--arms", nargs="+", default=["none", "snap"],
                help="distilled arms; `none` is also read snapped after distillation")
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--embedders", nargs="+", default=["ln_res_l", "dcnv2_base"])
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/snap_vs_tame"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

csv = os.path.join(a.out, "results.csv")
rows, hecho = reanudar(csv, ["dataset", "embedder", "ipc", "brazo", "seed"])
for ds in a.datasets:
    print(f"##### {ds} | {a.arms}", flush=True)
    for emb in a.embedders:
        for ipc in a.ipcs:
            for etq in a.arms:
                for run in range(a.runs):
                    seed = 132 + run
                    if (ds, emb, ipc, etq, seed) in hecho:
                        continue
                    nuevas = correr(ds, emb, ipc, etq, seed, a.iters, posthoc_snap=(etq == "none"),
                                    medir_normalidad=True)
                    for r in nuevas:
                        r["run"] = run
                    rows += nuevas
                    print(linea(nuevas[:3]) + f" | rejilla {nuevas[0]['en_rejilla']:.3f}", flush=True)
                    pd.DataFrame(rows).to_csv(csv, index=False)

df = pd.DataFrame(rows)
df.to_csv(csv, index=False)
pd.set_option("display.width", 240)
print("\n=== test accuracy, lowest-loss set, mean per arm ===")
print(df.pivot_table(index=["bloque", "classifier"], columns="brazo",
                     values="acc_bestloss").round(4).to_string())
print("\n=== share of the returned values that are observed values ===")
print(df.groupby("brazo").en_rejilla.mean().round(4).to_string())
print(f"\nCSV -> {csv}")
