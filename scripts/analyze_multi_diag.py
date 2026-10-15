"""Read the trajectories of `run_multi_diag.py` against the four candidate causes.

Every quantity is averaged over seeds per dataset and iteration, and shown as
its value minus the initialisation's, so 0 means "what random real rows give".
Trees is the mean of rf and xgboost on validation.

  A  `arboles` along the run: a steady fall from the init points to movement
     itself being harmful
  B  `pureza` and `margen` up, `frontera` down: the points left the class
     boundaries
  C  `spearman` up while `w1` stays flat: the joint structure drifted while
     the marginals did not
  D  the same columns across datasets, ordered by `cobertura`, the share of
     each class the init already holds
  E  `en_rejilla` dropping at once with the trees, and the snap table: trees
     on the returned set and the last iterate with every value snapped to the
     nearest one observed in its column, against the same sets unsnapped

The last table correlates, per dataset and seed, the change in tree accuracy
from the init to the last iterate with the change in each measure, which says
which of them moves together with the loss.

    .venv/bin/python scripts/analyze_multi_diag.py ~/tame_runs/multi_diag/results.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

MEDIDAS = ["arboles", "pureza", "frontera", "margen", "diversidad", "spearman", "w1", "sw",
           "en_rejilla", "distintos", "desplazamiento"]


def cargar(csv):
    d = pd.read_csv(csv)
    d["arboles"] = d[["val_rf", "val_xgboost"]].mean(1)
    d["arboles_test"] = d[["acc_rf", "acc_xgboost"]].mean(1)
    if "val_rf_snap" in d:
        d["arboles_snap"] = d[["val_rf_snap", "val_xgboost_snap"]].mean(1)
        d["arboles_test_snap"] = d[["acc_rf_snap", "acc_xgboost_snap"]].mean(1)
    return d


def relativo(d):
    """Each measure minus the init of the same distillation."""
    k = ["dataset", "embedder", "ipc", "brazo", "seed"]
    cols = MEDIDAS + ["arboles_test"]
    ini = d[d.iteracion == -1][k + cols]
    r = d.merge(ini, on=k, suffixes=("", "_ini"))
    for c in cols:
        r[c] = r[c] - r[f"{c}_ini"]
    return r.drop(columns=[f"{c}_ini" for c in cols])


def analizar(csv):
    d = cargar(csv)
    r = relativo(d)
    out = os.path.join(os.path.dirname(os.path.abspath(csv)), "analisis")
    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 240)
    its = sorted(i for i in d.iteracion.unique() if i >= 0)
    marcas = [-1] + [i for i in its if i in (0, 100, 250, 500, 750, its[-1])] + [-2]
    orden = (d.groupby("dataset").cobertura.mean().sort_values().index.tolist())
    tray = r.groupby(["dataset", "ipc", "brazo", "iteracion"])[MEDIDAS + ["arboles_test"]].mean().reset_index()
    tray.to_csv(os.path.join(out, "trayectoria.csv"), index=False)

    print("=== D: the regime. Share of each class held by the init, and the init's own values ===")
    ini = d[d.iteracion == -1].groupby(["dataset", "ipc"])[
        ["cobertura", "num_classes", "arboles", "pureza", "frontera", "spearman", "w1"]].mean()
    print(ini.reindex(orden, level=0).round(3).to_string())

    for ds in orden:
        for (ipc, brazo), t in tray[tray.dataset == ds].groupby(["ipc", "brazo"]):
            t = t.set_index("iteracion").reindex(marcas).rename(index={-1: "init", -2: "devuelto"})
            print(f"\n=== {ds} | ipc {ipc} | {brazo} | minus the init (trees x100) ===")
            t = t[MEDIDAS + ["arboles_test"]].copy()
            t[["arboles", "arboles_test"]] *= 100
            print(t.round(3).to_string())

    if "arboles_snap" in d:
        ini_t = d[d.iteracion == -1].set_index(["dataset", "embedder", "ipc", "brazo", "seed"])
        s_ = d[d.arboles_snap.notna()].copy()
        s_["conjunto"] = np.where(s_.es_best, "devuelto", "ultimo")
        k = ["dataset", "embedder", "ipc", "brazo", "seed"]
        s_ = s_.merge(ini_t[["arboles", "arboles_test"]].rename(columns=lambda c: c + "_init"),
                      left_on=k, right_index=True)
        tab = s_.groupby(["dataset", "conjunto"]).agg(
            val_sin=("arboles", "mean"), val_snap=("arboles_snap", "mean"),
            val_init=("arboles_init", "mean"), test_sin=("arboles_test", "mean"),
            test_snap=("arboles_test_snap", "mean"), test_init=("arboles_test_init", "mean"))
        tab["snap_recupera"] = tab.val_snap - tab.val_sin
        tab.reset_index().to_csv(os.path.join(out, "snap.csv"), index=False)
        print("\n=== E: trees on the set as is, snapped to observed values, and the init ===")
        print(tab.reindex(orden, level=0).round(4).to_string())

    # Per distillation: change from init to the last iterate, and what moves with
    # the change in trees.
    fin = r[r.iteracion == its[-1]]
    filas = []
    for ds, g in [("todos", fin)] + list(fin.groupby("dataset")):
        f = dict(dataset=ds, n=len(g), arboles=100 * g.arboles.mean())
        for m in MEDIDAS[1:]:
            f[f"r_{m}"] = (stats.spearmanr(g.arboles, g[m]).statistic
                           if len(g) > 3 and np.ptp(g[m]) > 0 else np.nan)
        filas.append(f)
    corr = pd.DataFrame(filas)
    corr.to_csv(os.path.join(out, "correlaciones.csv"), index=False)
    print("\n=== change in trees (x100) from init to the last iterate, and its Spearman r with the "
          "change in each measure ===")
    print(corr.round(2).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    analizar(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tame_runs/multi_diag/results.csv"))
