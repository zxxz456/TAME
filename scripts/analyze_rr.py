"""Accuracy, relative loss and relative regret: TAME on rails against plain TAME.

The paper's two summary metrics (Eq. 14), computed from runs already done:

    RL = A_full - A_dist                       in points, lower is better
    RR = (A_full - A_dist) / (A_full - A_rand)  0 = full data, 1 = random rows

A_dist is `acc_bestloss` of each arm, A_rand is `acc_init` (the random real
rows the run started from, same IPC and seed, shared by both arms) and A_full
comes from `run_full_data.py`, matched on dataset, seed and classifier.

TAME on rails is whatever of ste, W1D (0.03) and the in-loop snap has columns
to act on, so it is a different arm in each group of datasets:

    sin categoricas   comb_sin_cat   W1D + snap         (no one-hot groups)
    mixtos            comb_mixtos    ste + W1D + snap
    solo categoricas  onehot_modes   ste                (no column outside a group;
                                                         splice has one, constant)

All three share the grid (IPC 10, 50, 200; ln_res_l and dcnv2_base; seeds
132-141) and the classifier settings. From onehot_modes only the raw read
(`eval_proj == "crudo"`) is used, which is what plain TAME returns.

Everything is formed per cell (group, dataset, embedder, IPC, classifier) from
the means over the ten seeds. RR divides by A_full - A_rand, which can be near
zero, so cells where it is below 0.5 points get no RR; summaries of RR use the
median, as the paper does. Writes `<out>/celdas.csv`.

    .venv/bin/python scripts/analyze_rr.py
"""
import os, argparse
import numpy as np, pandas as pd

R = os.path.expanduser("~/tame_runs")
ap = argparse.ArgumentParser()
ap.add_argument("--full", default=f"{R}/full_data/results.csv")
ap.add_argument("--out", default=f"{R}/rr")
ap.add_argument("--min-gap", type=float, default=0.005,
                help="smallest A_full - A_rand (as a fraction) for RR to count")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

SOLO_CAT = ["phishing", "car", "kropt", "nursery5", "splice"]
FUENTES = [("sin categoricas", f"{R}/comb_sin_cat/results.csv", "w1@0.03+snap", None),
           ("mixtos", f"{R}/comb_mixtos/results.csv", "ste+w1@0.03+snap", None),
           ("solo categoricas", f"{R}/onehot_modes/results.csv", "ste", SOLO_CAT)]
CELDA = ["grupo", "dataset", "embedder", "ipc", "classifier"]

partes = []
for grupo, csv, rails, solo in FUENTES:
    d = pd.read_csv(csv)
    if solo is not None:
        d = d[d.dataset.isin(solo)]
    if "eval_proj" in d:
        d = d[d.eval_proj == "crudo"]
    d = d[d.brazo.isin(["none", rails])]
    d = d.assign(grupo=grupo, metodo=np.where(d.brazo == "none", "tame", "rails"))
    partes.append(d[CELDA + ["metodo", "seed", "acc_bestloss", "acc_init"]])
df = pd.concat(partes, ignore_index=True)

full = pd.read_csv(a.full)[["dataset", "seed", "classifier", "acc_full"]]
df = df.merge(full, on=["dataset", "seed", "classifier"], how="left")
falta = sorted(df[df.acc_full.isna()].dataset.unique())
if falta:
    raise SystemExit(f"no full-data score for {falta}: run run_full_data.py on them first")

c = (df.groupby(CELDA + ["metodo"])
       .agg(full=("acc_full", "mean"), dist=("acc_bestloss", "mean"), rand=("acc_init", "mean"),
            semillas=("seed", "nunique")).reset_index())
c["acc"] = 100 * c.dist
c["rl"] = 100 * (c.full - c.dist)
c["hueco"] = 100 * (c.full - c.rand)
c["rr"] = np.where(c.hueco >= 100 * a.min_gap, (c.full - c.dist) / (c.full - c.rand), np.nan)
c.to_csv(f"{a.out}/celdas.csv", index=False)

pd.set_option("display.width", 250)
print(f"{df.dataset.nunique()} datasets, {len(c) // 2} cells per method -> {a.out}/celdas.csv")
print(c.groupby(["grupo", "metodo"]).dataset.nunique().unstack().to_string())
t = (c.groupby(["grupo", "ipc", "classifier", "metodo"])
       .agg(acc=("acc", "mean"), rl=("rl", "mean"), rr=("rr", "median")).unstack("metodo"))
t.columns = [f"{k}_{m}" for k, m in t.columns]
print(t[[f"{k}_{m}" for k in ("acc", "rl", "rr") for m in ("tame", "rails")]].round(3).to_string())
sin_rr = c[c.rr.isna() & (c.metodo == "tame")].groupby("dataset").size()
print(f"\ncells with no RR (A_full - A_rand < {100 * a.min_gap:.1f} points):",
      sin_rr.to_dict() if len(sin_rr) else "none")
