"""Which columns does snapping help or hurt? A follow-up on the multi-class diagnosis.

`run_multi_diag.py` found that tree accuracy drops at the first SGD step, when
every value of the distilled set stops being a value observed in the real data
(hypothesis E, broken ties), and that snapping every value back to the nearest
observed value of its column brings `letter` back to the random-rows level,
`segment` only partly, and costs `credit` 4 points. Two questions were left:
does the rest of `segment` come back if the right columns are snapped, and where
does the loss of `credit` come from?

No distillation here. It reloads the sets the diagnosis saved
(`<out>/<dataset>/conjuntos/*.pt`, IPC 200, ln_res_l, seeds 2000-2004) and fits
rf and xgboost on each of them snapped five ways:

  sin redondear  as distilled
  todas          every column to its nearest observed training value
  discretas      only the columns with at most `--max-unicos` distinct values
                 (letter: all 16; segment: 3; credit: 9, sex, education,
                 marriage and the six PAY_x; spambase: none)
  continuas      only the other columns
  solo moda      a value is snapped only when its nearest observed value is the
                 column's most common one. spambase has no discrete column but
                 a single value, 0, holds 50 to 99% of the rows in most of them:
                 its ties are at zero, and this variant restores only those

on the returned set and on the last iterate, plus the initialisation (real rows)
as the reference. Validation and test accuracy are stored side by side.

    .venv/bin/python scripts/run_snap_columnas.py --datasets letter segment spambase credit
"""
import os, sys, glob, argparse, contextlib, io
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/zxxz6/TAME"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/zxxz6/TAME")
from w1_common import DEV, prepare_db, fit_eval

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["letter", "segment", "spambase", "credit"])
ap.add_argument("--diag", default=os.path.expanduser("~/tame_runs/multi_diag"),
                help="the output folder of run_multi_diag.py")
ap.add_argument("--max-unicos", type=int, default=20, help="a column with at most this many values is discrete")
ap.add_argument("--out", default=None, help="defaults to <diag>/snap_columnas/<dataset>.csv")
a = ap.parse_args()
ARB = ["rf", "xgboost"]


def rejillas(X_train):
    """Per column: the sorted observed values, the most common one, and whether it is discrete."""
    X = X_train.cpu().numpy()
    out = []
    for j in range(X.shape[1]):
        vals, cnt = np.unique(X[:, j], return_counts=True)
        out.append((vals, vals[cnt.argmax()], len(vals) <= a.max_unicos))
    return out


def redondear(X, rej, cuales):
    """Snap the chosen columns of X to their nearest observed value."""
    X = X.detach().cpu().numpy().copy()
    for j, (g, moda, discreta) in enumerate(rej):
        if cuales == "discretas" and not discreta or cuales == "continuas" and discreta:
            continue
        if len(g) == 1:
            near = np.full(len(X), g[0])
        else:
            i = np.clip(np.searchsorted(g, X[:, j]), 1, len(g) - 1)
            izq, der = g[i - 1], g[i]
            near = np.where(np.abs(X[:, j] - izq) <= np.abs(der - X[:, j]), izq, der)
        X[:, j] = np.where(near == moda, near, X[:, j]) if cuales == "solo moda" else near
    return torch.as_tensor(X)


VARIANTES = ["sin redondear", "todas", "discretas", "continuas", "solo moda"]
for ds in a.datasets:
    archivos = sorted(glob.glob(f"{a.diag}/{ds}/conjuntos/*.pt"))
    out = a.out or os.path.join(a.diag, "snap_columnas")
    os.makedirs(out, exist_ok=True)
    rows = []
    for f in archivos:
        t = torch.load(f)
        seed = int(t["cfg"]["random_seed"]); iters = max(k for k in t["snapshots"])
        with contextlib.redirect_stdout(io.StringIO()):
            data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
        rej = rejillas(data["X_train"])
        n_disc = sum(d for _, _, d in rej)
        y = t["y"]
        conjuntos = [("init", t["snapshots"][-1], ["sin redondear"]),
                     ("devuelto", t["best"], VARIANTES), ("ultimo", t["snapshots"][iters], VARIANTES)]
        for conj, X, variantes in conjuntos:
            for v in variantes:
                Xv = X if v == "sin redondear" else redondear(X, rej, v)
                fila = dict(dataset=ds, seed=seed, conjunto=conj, variante=v,
                            n_discretas=n_disc, n_columnas=len(rej))
                for clf in ARB:
                    fila[f"val_{clf}"], fila[f"acc_{clf}"], _ = fit_eval(Xv, y, data, clf, t["cfg"] | {"device": DEV}, seed)
                rows.append(fila)
        r = [x for x in rows if x["seed"] == seed and x["conjunto"] == "devuelto"]
        print(f"[{ds:9s} s{seed}] discretas {n_disc}/{len(rej)} | arboles val devuelto: " + "  ".join(
            f"{x['variante']}={np.mean([x['val_rf'], x['val_xgboost']]):.3f}" for x in r), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"{ds}.csv"), index=False)
    print(f"CSV -> {os.path.join(out, ds + '.csv')}", flush=True)
