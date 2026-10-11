"""Tune penalty's gamma and soft's temperatures before the four-mode ablation.

`run_onehot_modes.py` compares six arms, and two of them carry a hyperparameter:
`penalty` its weight gamma, `soft` the start and end of its temperature anneal.
Running the ablation with whatever value happened to be the default would test
the default, not the mode. This picks the values first.

Two separate experiments, one per output directory, chosen with ``--sweep``.
Each compares one mode against itself across values of its own hyperparameter;
neither is read against the other or against the remaining arms, which is what
the ablation is for:

  penalty   gamma over a grid, everything else fixed
  soft      tau0 x tau1 over a grid, the anneal tau0 -> tau1 geometric

**The choice is made on validation, never on test.** Tuning by test accuracy
and then reporting the ablation's test accuracy would hand `penalty` and `soft`
an advantage no other arm gets: they would be the only two that were fitted
while looking at the exam. So every row stores validation and test side by side,
the summary ranks by validation, and the seeds default to 1000-1009, disjoint
from the ablation's 132-141. Different seeds mean different train, validation
and test splits, so the values the ablation inherits were never chosen on its
own data.

`penalty` returns rows that are not necessarily legal, so its sets are evaluated
raw and projected at evaluation time, as in the ablation; `soft` returns legal
rows by construction and is asserted, not trusted.

Four blocks, binary and multi-class, each with and without categorical
columns. On a dataset with no one-hot groups the loop degrades every mode to
`none`, so there every value of the hyperparameter runs the same code. Those
blocks are not tuned: they run only the two extreme values, and the difference
between those two, which is the same code under two labels, is the noise floor
the real blocks' differences have to clear. Running the whole grid there would
measure that same noise ten times over.

Resumes: rows already in ``results.csv`` are skipped, so an interrupted shard
picks up where it stopped.
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

# Four blocks. The "sin" ones have zero one-hot groups (checked with
# OneHotProjector), so every mode is a no-op there and they serve as the noise
# floor. electricity was left out of BIN_SIN: it carries one group of 7 dummies.
BIN_CON   = ["adult", "bank", "german", "phishing", "law_school_admissions"]
BIN_SIN   = ["magic", "spambase", "credit", "madelon", "higgs"]
MULTI_CON = ["kropt", "covertype", "splice", "car", "nursery5"]
MULTI_SIN = ["letter", "pendigits", "satimage", "segment", "pageblocks"]

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", required=True, choices=["penalty", "soft"])
ap.add_argument("--datasets", nargs="+", default=BIN_CON + BIN_SIN + MULTI_CON + MULTI_SIN)
# gamma 0 would be `none` and is left out. Measured on adult, ipc 10, 1000
# iterations, on the last iterate: drift 0.062 at 0, 0.048 at 1, 0.022 at 10,
# 0.012 at 30 and a floor of 0.0098 at 100, with 0% of entries exact at every
# weight. The grid spans that descent. 300 is left out: on the last iterate it
# adds nothing over 100.
ap.add_argument("--gammas", nargs="+", type=float, default=[1.0, 3.0, 10.0, 30.0, 100.0])
ap.add_argument("--tau0s", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                help="soft: starting temperatures")
ap.add_argument("--tau1s", nargs="+", type=float, default=[0.01, 0.05, 0.2],
                help="soft: final temperatures; a pair is kept only if tau1 < tau0")
ap.add_argument("--gammas-nulos", nargs="+", type=float, default=[1.0, 100.0],
                help="penalty values run on datasets with no groups: the extremes only")
ap.add_argument("--taus-nulos", nargs="+", default=["0.5:0.01", "2:0.2"],
                help="soft tau0:tau1 pairs run on datasets with no groups: the extremes only")
ap.add_argument("--ipcs", nargs="+", type=int, default=[50])
ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1010)),
                help="disjoint from the ablation's 132-141 on purpose")
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--embedders", nargs="+", default=["ln_res_l", "dcnv2_base"])
ap.add_argument("--out", default=None, help="defaults to ~/tame_runs/onehot_tune_<sweep>")
a = ap.parse_args()
a.out = a.out or os.path.expanduser(f"~/tame_runs/onehot_tune_{a.sweep}")
os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLFS = ["mlp", "rf", "xgboost"]
DISCRETOS = ("ste", "soft", "project")
NAN = float("nan")


def _openml_cat(nombre, version, min_class=20, max_rows=None):
    """An OpenML dataset with categoricals, through tdbench's preprocessing.

    Classes with fewer than `min_class` rows are dropped because the stratified
    split needs at least one row per class in every part
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
    "car":       _openml_cat("car", 1),
    "nursery5":  _openml_cat("nursery", 1),
    "splice":    _openml_cat("splice", 1),
    "covertype": _openml_cat("covertype", 3, max_rows=100_000),
    "kropt":     _openml_cat("kropt", 1, min_class=60),
})
BLOQUE = {**{d: "binario-con" for d in BIN_CON}, **{d: "binario-sin" for d in BIN_SIN},
          **{d: "multi-con" for d in MULTI_CON}, **{d: "multi-sin" for d in MULTI_SIN}}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def brazos(tiene_grupos):
    """The arms of this sweep as (mode, gamma, tau0, tau1, label).

    The full grid where there are groups to act on, the two extremes where there
    are none, since there every value is the same code
    """
    if a.sweep == "penalty":
        gs = a.gammas if tiene_grupos else a.gammas_nulos
        return [("penalty", g, NAN, NAN, f"penalty@{g:g}") for g in gs]
    if tiene_grupos:
        pares = [(t0, t1) for t0 in a.tau0s for t1 in a.tau1s if t1 < t0]
    else:
        pares = [tuple(float(x) for x in p.split(":")) for p in a.taus_nulos]
    return [("soft", NAN, t0, t1, f"soft@{t0:g}-{t1:g}") for t0, t1 in pares]


def fit_eval(X_syn, y_syn, data, clf, cfg, seed):
    """Train one classifier and read it on validation and on test.

    Validation is what the sweep chooses on; test is stored next to it only so
    the notebook can show how far the two agree. Same model for both reads
    """
    set_seed(seed)
    td = {"X_train": X_syn, "y_train": y_syn, "X_val": data["X_val"], "y_val": data["y_val"],
          "input_dim": data["input_dim"], "num_classes": data["num_classes"]}
    with contextlib.redirect_stdout(io.StringIO()):
        m = train_classifier(td, {**cfg, "classifier": clf, "classifier_hidden": [128, 64],
                                  "classifier_epochs": 20})
        vista_val = dict(data); vista_val["X_test"] = data["X_val"]; vista_val["y_test"] = data["y_val"]
        val, _ = evaluate_classifier(m, vista_val, DEV)
        acc, auc = evaluate_classifier(m, data, DEV)
    return float(val), float(acc), (float(auc) if auc is not None and np.isfinite(auc) else np.nan)


csv = os.path.join(a.out, "results.csv")
prev = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
rows = prev.to_dict("records")
CLAVE = ["dataset", "embedder", "ipc", "brazo", "seed"]
hecho = set(map(tuple, prev[CLAVE].drop_duplicates().values)) if len(prev) else set()
if hecho:
    print(f"reanudando: {len(hecho)} destilaciones ya en {csv}", flush=True)

for ds in a.datasets:
  # One probe load decides how many arms this dataset gets. Cheap next to the grid.
  with contextlib.redirect_stdout(io.StringIO()):
      _sonda = prepare_db({"random_seed": a.seeds[0], "device": "cpu"}, name=ds)
  tiene = bool(OneHotProjector(_sonda["X_train"]))
  del _sonda
  lista = brazos(tiene)
  print(f"##### {ds} | {BLOQUE.get(ds, '?')} | {'con' if tiene else 'sin'} grupos | "
        f"{len(lista)} brazos -> {[b[-1] for b in lista]}", flush=True)
  for emb in a.embedders:
    for ipc in a.ipcs:
      for mode, gamma, tau0, tau1, etq in lista:
        for seed in a.seeds:
            if (ds, emb, ipc, etq, seed) in hecho:
                continue
            set_seed(seed)
            data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
            P = OneHotProjector(data["X_train"])
            cfg = {"device": DEV, "ipc": ipc, "dm_iters": a.iters, "dm_lr": 0.5,
                   "dm_batch_real": 128, "dm_embedder_type": emb,
                   "dm_embedder_size": "base", "dm_embed_hidden": 256,
                   "dm_embed_dim": max(4, min(48, ipc - 2)),
                   "random_seed": seed, "init_seed": seed,
                   "snapshot_every": a.iters, "return_snapshots": True,
                   "dm_onehot_mode": mode}
            if mode == "penalty":
                cfg["dm_onehot_gamma"] = gamma
            if mode == "soft":
                cfg["dm_onehot_tau0"], cfg["dm_onehot_tau1"] = tau0, tau1
            t0 = time.time()
            X_syn, y_syn, snaps = synthesize("tame", data, cfg)
            dt = time.time() - t0
            drift, fuera = P.drift(X_syn)
            snap = {it: X for it, X in snaps}
            # The returned set is chosen by lowest loss, and for `penalty` that
            # choice is biased towards the initialisation, where the penalty is
            # zero by construction. Its drift can then read 0 without the penalty
            # having done anything, so the last iterate is measured too, and the
            # fallback is recorded as a flag rather than left to be inferred.
            drift_fin, fuera_fin = P.drift(snap[a.iters])
            best_en_init = bool(torch.equal(X_syn, snap[-1]))
            if mode in DISCRETOS:
                assert torch.equal(P.hard(X_syn), X_syn), f"{etq} devolvio filas no legales"

            cands = [("bestloss", X_syn), ("final", snap[a.iters]), ("init", snap[-1])]
            # Discrete modes return legal rows, so projecting them again is the
            # identity by the assert above and would only repeat the same fits.
            vistas = {"crudo": lambda X: X}
            if mode not in DISCRETOS and P:
                vistas["proyectado"] = P.hard
            t1 = time.time()
            for vista, fn in vistas.items():
                proy = [(tag, fn(Xc)) for tag, Xc in cands]
                for clf in CLFS:
                    r = {tag: fit_eval(Xc, y_syn, data, clf, cfg, seed) for tag, Xc in proy}
                    rows.append(dict(
                        dataset=ds, bloque=BLOQUE.get(ds, "?"),
                        categoricas=("con" if P else "sin"),
                        tipo=("binario" if data["num_classes"] == 2 else "multiclase"),
                        num_classes=data["num_classes"], input_dim=data["input_dim"],
                        n_grupos=P.n_groups, n_dummy=P.n_dummy,
                        ipc=ipc, embedder=emb, sweep=a.sweep, mode=mode, brazo=etq,
                        gamma=gamma, tau0=tau0, tau1=tau1, eval_proj=vista,
                        seed=seed, classifier=clf,
                        val_bestloss=r["bestloss"][0], acc_bestloss=r["bestloss"][1],
                        auc_bestloss=r["bestloss"][2],
                        val_final=r["final"][0], acc_final=r["final"][1],
                        val_init=r["init"][0], acc_init=r["init"][1],
                        drift=drift, frac_fuera=fuera,
                        drift_final=drift_fin, frac_fuera_final=fuera_fin,
                        best_en_init=best_en_init,
                        synth_s=round(dt, 1), eval_s=round(time.time() - t1, 1)))
            ult = [x for x in rows[-len(CLFS) * len(vistas):] if x["eval_proj"] == "crudo"]
            print(f"[{ds:10s} {emb:10s} ipc{ipc:3d} {etq:16s} s{seed}] {dt:5.1f}s "
                  f"drift {drift:.4f}/{drift_fin:.4f}{' INIT' if best_en_init else ''} | " + "  ".join(
                f"{x['classifier']}={x['val_bestloss']:.3f}/{x['acc_bestloss']:.3f}" for x in ult),
                flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False)

df = pd.DataFrame(rows)
df.to_csv(csv, index=False)
pd.set_option("display.width", 240)

# The ranking that picks the value. Validation only; test sits next to it so the
# notebook can show whether the two orderings agree, not to choose with.
# Two rankings, lowest-loss set and last iterate, on the blocks with groups only:
# elsewhere every value is the same code. `best_en_init` is the share of runs whose
# returned set was the initialisation; with penalty out of the selection
# (tame_synth.py) it should stay at zero, and it is kept as the check that it does.
con = df[df.categoricas == "con"]
t = con[con.classifier.isin(["rf", "xgboost"])]
if len(t):
    print("\n=== con categoricas; media sobre datasets y semillas; por VALIDACION, arboles ===")
    for vista in sorted(t.eval_proj.unique()):
        for emb in sorted(t.embedder.unique()):
            s_ = (t[(t.eval_proj == vista) & (t.embedder == emb)]
                  .groupby("brazo")[["val_bestloss", "acc_bestloss", "val_final", "acc_final",
                                     "drift", "drift_final", "best_en_init"]].mean())
            if len(s_) > 1:
                print(f"\n-- {emb} | {vista} --")
                print(s_.sort_values("val_bestloss", ascending=False).round(4).to_string())
                print(f"   elegido por validacion, conjunto de menor perdida: "
                      f"{s_.val_bestloss.idxmax()}")
                print(f"   elegido por validacion, ultimo iterado:            "
                      f"{s_.val_final.idxmax()}")
    m = con[(con.classifier == "mlp") & (con.eval_proj == "crudo")]
    print("\n=== lo mismo para mlp, validacion ===")
    print(m.pivot_table(index="brazo", columns="embedder", values="val_bestloss")
          .round(4).to_string())

# The noise floor. On the blocks without groups the two extremes are the same
# code under two labels, so their paired difference is what chance alone
# produces. A difference between two values on the real blocks that is not
# clearly larger than this is not a difference.
sin = df[(df.categoricas == "sin") & (df.eval_proj == "crudo")]
if sin.brazo.nunique() == 2:
    b0, b1 = sorted(sin.brazo.unique())
    pv = sin.pivot_table(index=["dataset", "embedder", "ipc", "seed", "classifier"],
                         columns="brazo", values="val_bestloss")
    d = (pv[b1] - pv[b0]).dropna()
    print(f"\n=== piso de ruido: {b1} menos {b0} donde los dos son el mismo codigo ===")
    print(d.groupby(level="classifier").agg(["mean", "std", lambda x: x.abs().mean()])
          .rename(columns={"<lambda_0>": "media_abs"}).round(4).to_string())

print(f"\nCSV -> {csv}")
