"""TAME with one-hot inside the loop against plain TAME.

Two arms on shared seeds and shared initialisations:

  none   the published loop: one-hot dummies are optimised as free reals and
         end up outside their two legal values
  ste    the one-hot projection inside the loop. The embedder sees genuine
         one-hot rows and the gradient passes straight to the continuous
         variable, so moments are compared between legal rows on both sides

Accuracy of the three classifiers is measured on three sets: the one
distillation returns (lowest loss), iterate 1000, and the initialisation, which
is real rows and serves as the yardstick.

All ten datasets have genuine one-hot columns. That matters: on a purely numeric
dataset the projector has nothing to touch, the `ste` arm runs the same code as
`none` and the comparison says nothing. That is why `letter`, sixteen numeric
columns and zero categoricals, was swapped for `kropt`, which has seventeen
classes and 100% of its columns in one-hot groups.
"""
import os, sys, time, random, argparse, contextlib, io
import numpy as np, pandas as pd, torch
from sklearn.datasets import fetch_openml
sys.path.insert(0, "/home/zxxz6/TAME"); os.chdir("/home/zxxz6/TAME")
from data.prepare_database import prepare_db, DATASET_REGISTRY
from data.tdbench_datasets import register_tdbench_datasets, _preprocess_and_split
from synth.registry import synthesize
from synth.onehot import OneHotProjector
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier

BINARIOS = ["adult", "bank", "german", "phishing", "law_school_admissions"]
MULTI    = ["kropt", "covertype", "splice", "car", "nursery5"]

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=BINARIOS + MULTI)
ap.add_argument("--modes", nargs="+", default=["none", "ste"])
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--embedders", nargs="+", default=["ln_res_l", "dcnv2_base"])
ap.add_argument("--gamma", type=float, default=100.0)
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/onehot_vs_tame"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLFS = ["mlp", "rf", "xgboost"]
DISCRETOS = ("ste", "soft", "project")


def _openml_cat(nombre, version, min_class=20, max_rows=None):
    """An OpenML dataset with categoricals, through tdbench's preprocessing.

    Classes with fewer than `min_class` rows are dropped because the stratified
    split needs at least one row per class in every part, and at a high IPC a
    class of twenty rows would be sampled with replacement almost in full
    """
    def prep(random_seed=42, device="cpu"):
        ds = fetch_openml(nombre, version=version, as_frame=True)
        X_df = ds.data
        y = pd.Series(np.asarray(ds.target).ravel()).astype(str)
        keep = y.map(y.value_counts()).astype(int) >= min_class
        X_df, y = X_df[keep.values].reset_index(drop=True), y[keep.values].reset_index(drop=True)
        if max_rows and len(y) > max_rows:
            idx = np.random.default_rng(0).choice(len(y), max_rows, replace=False)
            X_df, y = X_df.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        print(f"{nombre}: {len(y)} filas, {y.nunique()} clases")
        return _preprocess_and_split(X_df, y, random_seed, device)
    return prep


register_tdbench_datasets()
DATASET_REGISTRY.update({
    "car":       _openml_cat("car", 1),                           # 4 clases
    "nursery5":  _openml_cat("nursery", 1),                       # 4 tras filtrar
    "splice":    _openml_cat("splice", 1),                        # 3 clases
    "covertype": _openml_cat("covertype", 3, max_rows=100_000),   # 7 clases
    # 17 classes and 100% of the columns in one-hot groups. Replaces letter,
    # which has not one categorical and would leave both arms identical.
    "kropt":     _openml_cat("kropt", 1, min_class=60),
})


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def fit_eval(X_syn, y_syn, data, clf, cfg, seed):
    """Entrena un clasificador con el conjunto destilado y lo mide en test."""
    set_seed(seed)
    td = {"X_train": X_syn, "y_train": y_syn, "X_val": data["X_val"], "y_val": data["y_val"],
          "input_dim": data["input_dim"], "num_classes": data["num_classes"]}
    with contextlib.redirect_stdout(io.StringIO()):
        m = train_classifier(td, {**cfg, "classifier": clf, "classifier_hidden": [128, 64],
                                  "classifier_epochs": 20})
        acc, auc = evaluate_classifier(m, data, DEV)
    return float(acc), (float(auc) if auc is not None and np.isfinite(auc) else np.nan)


rows = []
csv = os.path.join(a.out, "results.csv")
for ds in a.datasets:
  for emb in a.embedders:
    for ipc in a.ipcs:
      for mode in a.modes:
        print(f"\n===== {ds} | {emb} | ipc {ipc} | {mode} | {a.runs} corridas =====", flush=True)
        for run in range(a.runs):
            seed = 132 + run
            set_seed(seed)
            data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
            P = OneHotProjector(data["X_train"])
            cfg = {"device": DEV, "ipc": ipc, "dm_iters": a.iters, "dm_lr": 0.5,
                   "dm_batch_real": 128, "dm_embedder_type": emb,
                   "dm_embedder_size": "base", "dm_embed_hidden": 256,
                   "dm_embed_dim": max(4, min(48, ipc - 2)),
                   "random_seed": seed, "init_seed": seed,
                   # -1 = initialisation (real rows), 0 and 1000 = iterates
                   "snapshot_every": a.iters, "return_snapshots": True,
                   "dm_onehot_mode": mode, "dm_onehot_gamma": a.gamma}
            t0 = time.time()
            X_syn, y_syn, snaps = synthesize("tame", data, cfg)
            dt = time.time() - t0
            drift, fuera = P.drift(X_syn)
            if mode in DISCRETOS and P:
                assert torch.equal(P.hard(X_syn), X_syn), f"{mode} devolvio filas no legales"

            snap = {it: X for it, X in snaps}
            cands = [("bestloss", X_syn), ("final", snap[a.iters]), ("init", snap[-1])]
            for clf in CLFS:
                accs = {}
                for tag, Xc in cands:
                    accs[tag] = fit_eval(Xc, y_syn, data, clf, cfg, seed)
                rows.append(dict(
                    dataset=ds, tipo=("binario" if data["num_classes"] == 2 else "multiclase"),
                    num_classes=data["num_classes"], input_dim=data["input_dim"],
                    n_grupos=P.n_groups, n_dummy=P.n_dummy, ipc=ipc,
                    embedder=emb, mode=mode,
                    run=run, seed=seed, classifier=clf,
                    acc_bestloss=accs["bestloss"][0], auc_bestloss=accs["bestloss"][1],
                    acc_final=accs["final"][0], auc_final=accs["final"][1],
                    acc_init=accs["init"][0], auc_init=accs["init"][1],
                    drift=drift, frac_fuera=fuera, synth_s=round(dt, 1)))
            print(f"[{ds:10s} {emb:10s} ipc{ipc:3d} {mode:5s} run{run:02d}] {dt:5.1f}s "
                  f"drift {drift:.3f} | " + "  ".join(
                f"{r['classifier']}={r['acc_bestloss']:.4f}" for r in rows[-3:]), flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False)

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\n=== accuracy media sobre las corridas (conjunto de menor perdida) ===")
print(df.pivot_table(index=["tipo", "dataset", "embedder", "ipc", "classifier"], columns="mode",
                     values="acc_bestloss").round(4).to_string())
if len(a.modes) == 2:
    from scipy import stats
    p = df.pivot_table(index=["tipo", "dataset", "embedder", "ipc", "classifier", "run"],
                       columns="mode", values="acc_bestloss")
    p["delta"] = p[a.modes[1]] - p[a.modes[0]]
    out = []
    for k, g in p.groupby(level=[0, 1, 2, 3, 4]):
        x = g["delta"].dropna().values
        pw = stats.wilcoxon(x)[1] if len(x) > 5 and np.ptp(x) > 0 else np.nan
        out.append((*k, len(x), x.mean(), np.median(x), int((x > 0).sum()), pw))
    print(f"\n=== delta pareado ({a.modes[1]} menos {a.modes[0]}), por semilla ===")
    print(pd.DataFrame(out, columns=["tipo", "dataset", "embedder", "ipc", "clasificador", "n",
                                     "media", "mediana", "gana", "wilcoxon"])
          .round(4).to_string(index=False))
    print(f"\n=== resumen por embedder, presupuesto y clasificador "
          f"({a.modes[1]} menos {a.modes[0]}) ===")
    r = p.reset_index().groupby(["embedder", "ipc", "classifier"]).delta
    print(pd.DataFrame({"media": r.mean(), "mediana": r.median(),
                        "gana": r.apply(lambda x: int((x > 0).sum())),
                        "de": r.size()}).round(4).to_string())
print("\n=== deriva de lo devuelto, por modo (0 = sigue siendo one-hot) ===")
print(df.drop_duplicates(["dataset", "embedder", "ipc", "mode", "run"])
        .groupby("mode")[["drift", "frac_fuera"]]
        .mean().round(4).to_string())
print("\nCSV ->", csv)
