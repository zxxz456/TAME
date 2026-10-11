"""Plain TAME against TAME with `ste`, read from the ablation's results.

The ablation (`run_onehot_modes.py`) ran five arms. This keeps two of them:
`none`, the published loop, and `ste`, the in-loop one-hot projection the
ablation picked, and lays them side by side at three levels of detail:

  por_dataset  one row per dataset, IPC and classifier
  por_ipc      one row per block, IPC and classifier, averaging its datasets
  general      one row per scope and classifier, averaging datasets and IPCs;
               the scopes are the four blocks, "con categoricas" (the two blocks
               with one-hot groups), "sin categoricas" (the two without) and all
               twenty datasets

Every row pairs each distillation with its twin: same dataset, embedder, IPC,
seed and classifier. Both embedders go into the same cell, so a dataset and IPC
cell holds 20 pairs (2 embedders x 10 seeds). Columns: `tame` and `ste` are the
mean accuracies, `delta` the mean paired difference, `ic95` its 95% half-width,
`gana` how many pairs `ste` won, and `wilcoxon` the signed-rank p on the deltas.

The blocks are balanced, every dataset has the same number of pairs, so a mean
over pairs is also the macro mean over datasets.

On the blocks without categoricals the projector has nothing to touch and `ste`
runs the same code as `none`. The two differ only through the embedder draws,
which are seeded from the clock, so their delta is the noise floor of the
comparison and should sit at zero.

    .venv/bin/python scripts/ste_vs_tame.py                   # saves and prints
    .venv/bin/python scripts/ste_vs_tame.py --metric auc_bestloss
"""
import os, argparse
import numpy as np, pandas as pd
from scipy import stats

CSV = os.path.expanduser("~/tame_runs/onehot_modes/results.csv")
OUT = os.path.expanduser("~/tame_runs/onehot_modes/ste_vs_tame")

CLFS = ["rf", "xgboost", "mlp"]
IPCS = [10, 50, 200]
BLOQUES = ["binario-banco", "binario-sin-onehot", "multi-onehot", "multi-sin-onehot"]
NOM_BLOQUE = {"binario-banco": "binario con categoricas",
              "binario-sin-onehot": "binario sin categoricas",
              "multi-onehot": "multiclase con categoricas",
              "multi-sin-onehot": "multiclase sin categoricas"}
CON = ["binario-banco", "multi-onehot"]
SIN = ["binario-sin-onehot", "multi-sin-onehot"]
AMBITOS = [NOM_BLOQUE[b] for b in BLOQUES] + ["con categoricas", "sin categoricas", "los 20 datasets"]
LLAVE = ["bloque", "dataset", "embedder", "ipc", "classifier", "run"]


def cargar(csv=CSV, metric="acc_bestloss", embedder=None):
    """One row per pair: the `none` and `ste` accuracy of the same distillation.

    Only the raw evaluation view: `ste` returns legal rows, so its projected view
    is the same number, and `none` projected is the post-hoc fix, which is a
    different comparison"""
    d = pd.read_csv(csv)
    d = d[d.brazo.isin(["none", "ste"]) & (d.eval_proj == "crudo")]
    if embedder:
        d = d[d.embedder == embedder]
    p = d.pivot_table(index=LLAVE, columns="brazo", values=metric).reset_index()
    p = p.rename(columns={"none": "tame"}).dropna(subset=["tame", "ste"])
    p["delta"] = p.ste - p.tame
    p["categoricas"] = np.where(p.bloque.isin(CON), "con", "sin")
    return p


def _resumen(g):
    v = g.delta.values
    return pd.Series(dict(
        n=len(v), tame=g.tame.mean(), ste=g.ste.mean(), delta=v.mean(),
        ic95=1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
        gana=int((v > 0).sum()),
        wilcoxon=stats.wilcoxon(v)[1] if np.ptp(v) > 0 else np.nan))


def _orden(t):
    """Blocks, classifiers and IPCs in reading order, not alphabetical."""
    if "bloque" in t:
        t["bloque"] = pd.Categorical(t.bloque, BLOQUES, ordered=True)
    if "ambito" in t:
        t["ambito"] = pd.Categorical(t.ambito, AMBITOS, ordered=True)
    t["classifier"] = pd.Categorical(t.classifier, CLFS, ordered=True)
    by = [c for c in ("ambito", "bloque", "dataset", "ipc", "classifier") if c in t]
    return t.sort_values(by).reset_index(drop=True)


def por_dataset(p):
    """Per dataset, IPC and classifier: 20 pairs each."""
    t = p.groupby(["bloque", "dataset", "ipc", "classifier"], observed=True).apply(
        _resumen, include_groups=False).reset_index()
    return _orden(t)


def por_ipc(p):
    """Per block, IPC and classifier: 5 datasets x 20 pairs = 100 each."""
    t = p.groupby(["bloque", "ipc", "classifier"], observed=True).apply(
        _resumen, include_groups=False).reset_index()
    return _orden(t)


def general(p):
    """Per scope and classifier, every IPC together.

    A block holds 300 pairs per classifier, "con" and "sin" 600, all 1200"""
    partes = [p.assign(ambito=p.bloque.map(NOM_BLOQUE)),
              p[p.bloque.isin(CON)].assign(ambito="con categoricas"),
              p[p.bloque.isin(SIN)].assign(ambito="sin categoricas"),
              p.assign(ambito="los 20 datasets")]
    t = pd.concat(partes).groupby(["ambito", "classifier"], observed=True).apply(
        _resumen, include_groups=False).reset_index()
    return _orden(t)


def ancha(t, filas, cols=("tame", "ste", "delta")):
    """Long table to the reading layout: classifiers across, (tame, ste, delta)
    under each."""
    w = t.pivot_table(index=filas, columns="classifier", values=list(cols), observed=True)
    w = w.swaplevel(0, 1, axis=1)
    return w.reindex(columns=pd.MultiIndex.from_product([CLFS, list(cols)]))


def calcular(csv=CSV, metric="acc_bestloss", embedder=None):
    p = cargar(csv, metric, embedder)
    return {"por_dataset": por_dataset(p), "por_ipc": por_ipc(p), "general": general(p)}


def guardar(tablas, out=OUT, sufijo=""):
    os.makedirs(out, exist_ok=True)
    for nom, t in tablas.items():
        t.to_csv(os.path.join(out, f"{nom}{sufijo}.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV, help="the merged results.csv of run_onehot_modes.py")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--metric", default="acc_bestloss",
                    choices=["acc_bestloss", "acc_final", "auc_bestloss", "auc_final"])
    a = ap.parse_args()
    tablas = calcular(a.csv, a.metric)
    sufijo = "" if a.metric == "acc_bestloss" else f"_{a.metric}"
    guardar(tablas, a.out, sufijo)
    pd.set_option("display.width", 200)
    print(f"{a.metric} -> {a.out}\n")
    print(ancha(tablas["general"], "ambito").round(4).to_string())
