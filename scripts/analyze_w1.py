"""TAME with the W1 marginal term against plain TAME, from `run_w1_vs_tame.py`.

Every arm is paired with `none` on the same dataset, embedder, IPC, seed and
classifier, and the tables follow the ones the `ste` comparison was read with,
so the two can be put side by side:

  por_dataset      per dataset, IPC and classifier: none, the arm, the paired
                   difference, its 95% half-width, wins and Wilcoxon p
  por_ipc          the same per block and IPC
  general          per scope (each block and all datasets) and classifier
  vs_filas_reales  accuracy minus that of the run's own initialisation, which is
                   real rows; trees only, `none` included. The target: in
                   multi-class, plain TAME falls below random rows at IPC 50
                   and 200
  mecanismo        W1 to the real marginals and Shapiro-Wilk of the
                   initialisation, the returned set and the last iterate, per
                   arm, block and IPC. If the term works for the stated reason,
                   its sets sit closer to the real marginals and are less
                   Gaussian than plain TAME's
  final_vs_best    last iterate minus lowest-loss set. The Gaussianisation
                   account says the end of the run is where trees suffer; with
                   the term that gap should close

Written to ``<csv dir>/analisis/``. `run_sharded.sh` runs it after merging with
``POST=analyze_w1.py``; by hand:

    .venv/bin/python scripts/analyze_w1.py ~/tame_runs/w1_vs_tame/results.csv
"""
import os, argparse
import numpy as np, pandas as pd
from scipy import stats

CLFS = ["rf", "xgboost", "mlp"]
ARB = ["rf", "xgboost"]
LLAVE = ["bloque", "dataset", "embedder", "ipc", "classifier", "seed"]


def _resumen(g):
    v = g.delta.values
    return pd.Series(dict(
        n=len(v), tame=g.acc_none.mean(), brazo_acc=g.acc.mean(), delta=v.mean(),
        ic95=1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
        gana=int((v > 0).sum()),
        wilcoxon=stats.wilcoxon(v)[1] if np.ptp(v) > 0 else np.nan))


def pares(df, metric="acc_bestloss"):
    """One row per paired run and arm."""
    p = df.pivot_table(index=LLAVE, columns="brazo", values=metric)
    init = df[df.brazo == "none"].set_index(LLAVE)[metric.split("_")[0] + "_init"]
    out = []
    for arm in [c for c in p.columns if c != "none"]:
        q = p[[arm, "none"]].dropna().copy()
        q.columns = ["acc", "acc_none"]
        q["acc_init"] = init.reindex(q.index)
        q["arm"] = arm
        out.append(q.reset_index())
    out = pd.concat(out, ignore_index=True)
    out["delta"] = out.acc - out.acc_none
    out["vs_reales"] = out.acc - out.acc_init
    return out


def _agrupa(pr, por):
    return (pr.groupby(["arm"] + por).apply(_resumen, include_groups=False)
            .reset_index())


def general(pr):
    partes = [pr.assign(ambito=pr.bloque), pr.assign(ambito="todos")]
    return _agrupa(pd.concat(partes), ["ambito", "classifier"])


def vs_filas_reales(df, pr, metric="acc_bestloss"):
    """Gain over random real rows, trees only, `none` included, per block and IPC."""
    base = df[(df.brazo == "none") & df.classifier.isin(ARB)]
    s_none = base.assign(arm="none", vs_reales=base[metric] - base[metric.split("_")[0] + "_init"])
    todo = pd.concat([s_none[["arm", "bloque", "ipc", "vs_reales"]],
                      pr[pr.classifier.isin(ARB)][["arm", "bloque", "ipc", "vs_reales"]]])
    return (todo.groupby(["arm", "bloque", "ipc"]).vs_reales
            .agg(n="size", media="mean",
                 le_gana=lambda v: int((v > 0.002).sum())).reset_index())


def mecanismo(df):
    """Marginal measures per arm, block and IPC; one row per distillation."""
    d = df.drop_duplicates(["dataset", "embedder", "ipc", "brazo", "seed"])
    medidas = [m for m in ("w1", "sw", "swp", "swc_pass", "swc_w", "mardia_rel", "mardia_dev")
               if f"{m}_bestloss" in d]
    cols = [f"{m}_{k}" for m in medidas for k in ("init", "bestloss", "final")]
    return d.groupby(["brazo", "bloque", "ipc"])[cols].mean().reset_index()


def final_vs_best(df):
    d = df.assign(gap=df.acc_final - df.acc_bestloss)
    return d.pivot_table(index=["brazo", "bloque"], columns="classifier",
                         values="gap").reindex(columns=CLFS).reset_index()


def analizar(csv, out=None, metric="acc_bestloss"):
    df = pd.read_csv(csv)
    out = out or os.path.join(os.path.dirname(os.path.abspath(csv)), "analisis")
    os.makedirs(out, exist_ok=True)
    pr = pares(df, metric)
    t = {"pares": pr,
         "por_dataset": _agrupa(pr, ["bloque", "dataset", "ipc", "classifier"]),
         "por_ipc": _agrupa(pr, ["bloque", "ipc", "classifier"]),
         "general": general(pr),
         "vs_filas_reales": vs_filas_reales(df, pr, metric),
         "mecanismo": mecanismo(df),
         "final_vs_best": final_vs_best(df)}
    sufijo = "" if metric == "acc_bestloss" else f"_{metric}"
    for nom, x in t.items():
        x.to_csv(os.path.join(out, f"{nom}{sufijo}.csv"), index=False)

    pd.set_option("display.width", 240)
    print(f"{len(df)} rows, {df.dataset.nunique()} datasets, metric {metric} -> {out}")
    g = t["general"]
    print("\n=== each arm minus plain TAME, mean over everything ===")
    print(g.pivot_table(index=["arm", "ambito"], columns="classifier", values="delta")
          .reindex(columns=CLFS).round(4).to_string())
    print("\n=== the same by IPC, trees ===")
    pi = t["por_ipc"]
    print(pi[pi.classifier.isin(ARB)].pivot_table(index=["arm", "bloque"], columns=["ipc", "classifier"],
                                                  values="delta").round(4).to_string())
    print("\n=== over random real rows, trees (mean) ===")
    print(t["vs_filas_reales"].pivot_table(index=["arm", "bloque"], columns="ipc",
                                           values="media").round(4).to_string())
    print("\n=== marginals: W1 to the real columns (lower = closer), SW (higher = more Gaussian) ===")
    print(t["mecanismo"].round(4).to_string(index=False))
    print("\n=== last iterate minus lowest-loss set ===")
    print(t["final_vs_best"].round(4).to_string(index=False))
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="the merged results.csv of run_w1_vs_tame.py")
    ap.add_argument("--out", default=None, help="defaults to <csv dir>/analisis")
    ap.add_argument("--metric", default="acc_bestloss",
                    choices=["acc_bestloss", "acc_final", "auc_bestloss", "auc_final"])
    a = ap.parse_args()
    analizar(a.csv, a.out, a.metric)
