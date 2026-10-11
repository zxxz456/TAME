"""The four in-loop one-hot modes against plain TAME, on shared seeds.

`run_onehot_vs_tame.py` compared the published loop against `ste` alone, because
`ste` was the one with a reason to work. This closes the ablation: a reviewer
asking "why that one and not projected gradient descent, which invents nothing"
should find the answer measured rather than argued.

Five arms, all from the same `dm_onehot_mode` flag:

  none          the published loop, dummies optimised as free reals
  ste           hard projection forwards, identity backwards
  soft          softmax per group, temperature annealed 0.5 -> 0.2
  project       projected gradient descent: SGD step, then projection in place
  penalty@10    L1 distance to the legal value added to the loss

The two hyperparameters come from `run_onehot_tuning.py`, chosen on validation
with seeds 1000-1009, disjoint from the 132-141 used here, so this experiment's
test numbers were never looked at while choosing them. Neither mattered much:
across gamma, and across nine (tau0, tau1) pairs, tree accuracy moved by less
than half a point. gamma 10 was a small but consistent winner once the output is
projected; for soft no pair was distinguishable and 0.5 -> 0.2 is simply the
validation-best. A second penalty arm was dropped for the same reason: a weight
that does not move the result does not need two arms. `penalty` never returns
legal rows (its drift floors near 0.01), which is why the double evaluation
below matters most for it.

Two things make this more than the previous sweep rerun with a longer list.

**Every distilled set is evaluated twice**, raw and with the projection applied
at evaluation time. `none` and `penalty` return rows that are not necessarily
legal, so judging them against three arms that return legal rows by construction
would conflate two questions: whether the constraint helped the optimisation,
and whether the output happened to be legal. Evaluating both ways separates
them, and it comes for free, since the distillation is already paid for. The
side effect is that `none` evaluated projected reproduces the post-hoc
experiment, so in-loop and post-hoc end up in one table instead of two.

**The null controls run only `none` and `ste`.** There are two, binary and
multi-class datasets with no categorical columns. On those the loop degrades any
mode to `none`, so the other three would run byte-identical code and differ only
by RNG; keeping them would have cost 1800 distillations to measure noise.

    THREADS=2 POST=analyze_onehot_modes.py bash scripts/run_sharded.sh \\
      run_onehot_modes.py ~/tame_runs/onehot_modes 8 <the twenty datasets>
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

# The three blocks of the previous sweep, kept identical so the two tables can be
# read side by side, plus a binary null control the previous sweep did not have,
# so both task types get one. The null blocks have no categorical columns (checked
# with OneHotProjector), so every mode is a no-op there by construction.
# electricity was not used for BIN_NULOS: it carries one group of 7 dummies.
BINARIOS  = ["adult", "bank", "german", "phishing", "law_school_admissions"]
BIN_NULOS = ["magic", "spambase", "credit", "madelon", "higgs"]
MULTI     = ["kropt", "covertype", "splice", "car", "nursery5"]
NULOS     = ["letter", "pendigits", "satimage", "segment", "pageblocks"]

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=BINARIOS + BIN_NULOS + MULTI + NULOS)
ap.add_argument("--modes", nargs="+",
                default=["none", "ste", "soft", "project", "penalty"])
ap.add_argument("--modes-nulos", nargs="+", default=["none", "ste"],
                help="modes run on datasets with no one-hot groups, where every "
                     "mode degrades to none and the rest would be duplicates")
# Chosen by run_onehot_tuning.py on validation, IPC 50, ~/tame_runs/onehot_tune_*.
ap.add_argument("--gammas", nargs="+", type=float, default=[10.0],
                help="one arm per weight, for penalty only")
ap.add_argument("--tau0", type=float, default=0.5, help="soft: starting temperature")
ap.add_argument("--tau1", type=float, default=0.2, help="soft: final temperature")
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--embedders", nargs="+", default=["ln_res_l", "dcnv2_base"])
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/onehot_modes"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLFS = ["mlp", "rf", "xgboost"]
# The modes whose returned rows are legal by construction, so the projection at
# evaluation time has to be the identity on them. That is asserted, not assumed.
DISCRETOS = ("ste", "soft", "project")


_BAJADOS = {}


def _bajar(nombre, version, intentos=6, espera=10):
    """fetch_openml by name, once per process, retried on network errors.

    By name, sklearn asks the OpenML API to resolve the name on every call, so a
    loader that runs once per seed made 300 network requests per dataset, and a
    single DNS hiccup at run 111 took `car` down. The raw frame does not depend on
    the seed (the split does, and that happens afterwards), so it is fetched once
    and kept; the retries cover the first fetch
    """
    k = (nombre, version)
    if k not in _BAJADOS:
        for i in range(intentos):
            try:
                _BAJADOS[k] = fetch_openml(nombre, version=version, as_frame=True)
                break
            except OSError as e:          # URLError, timeouts, name resolution
                if i == intentos - 1:
                    raise
                print(f"{nombre}: fallo la red ({e}); reintento en {espera * 2 ** i}s", flush=True)
                time.sleep(espera * 2 ** i)
    return _BAJADOS[k]


def _openml_cat(nombre, version, min_class=20, max_rows=None):
    """An OpenML dataset with categoricals, through tdbench's preprocessing.

    Classes with fewer than `min_class` rows are dropped because the stratified
    split needs at least one row per class in every part, and at a high IPC a
    class of twenty rows would be sampled with replacement almost in full
    """
    def prep(random_seed=42, device="cpu"):
        ds = _bajar(nombre, version)
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
    "kropt":     _openml_cat("kropt", 1, min_class=60),           # 17 clases
})

BLOQUE = {**{d: "binario-banco" for d in BINARIOS},
          **{d: "binario-sin-onehot" for d in BIN_NULOS},
          **{d: "multi-onehot" for d in MULTI},
          **{d: "multi-sin-onehot" for d in NULOS}}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def brazos(tiene_grupos):
    """The (mode, gamma) pairs to run, as one list.

    `penalty` becomes one arm per weight; everything else carries a NaN gamma so
    the column stays numeric. On a dataset with no groups the short list is used,
    because the rest would be the same code with another label
    """
    modos = a.modes if tiene_grupos else a.modes_nulos
    out = []
    for m in modos:
        if m == "penalty":
            out += [(m, g) for g in a.gammas]
        else:
            out.append((m, float("nan")))
    return out


def fit_eval(X_syn, y_syn, data, clf, cfg, seed):
    """Train one classifier on a distilled set and read test accuracy and AUC."""
    set_seed(seed)
    td = {"X_train": X_syn, "y_train": y_syn, "X_val": data["X_val"], "y_val": data["y_val"],
          "input_dim": data["input_dim"], "num_classes": data["num_classes"]}
    with contextlib.redirect_stdout(io.StringIO()):
        m = train_classifier(td, {**cfg, "classifier": clf, "classifier_hidden": [128, 64],
                                  "classifier_epochs": 20})
        acc, auc = evaluate_classifier(m, data, DEV)
    return float(acc), (float(auc) if auc is not None and np.isfinite(auc) else np.nan)


csv = os.path.join(a.out, "results.csv")
# Resume: distillations already in results.csv are skipped, so a shard that died
# halfway picks up where it stopped instead of overwriting what it had. Rows are
# written after each whole distillation, so anything in the file is complete.
prev = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
rows = prev.to_dict("records")
hecho = (set(map(tuple, prev[["dataset", "embedder", "ipc", "brazo", "run"]]
                 .drop_duplicates().values)) if len(prev) else set())
if hecho:
    print(f"reanudando: {len(hecho)} destilaciones ya en {csv}", flush=True)
for ds in a.datasets:
  # One probe load to find out whether this dataset has one-hot groups at all,
  # which decides how many arms it gets. Cheap next to the grid below.
  _probe = prepare_db({"random_seed": 132, "device": DEV}, name=ds)
  tiene = bool(OneHotProjector(_probe["X_train"]))
  del _probe
  lista = brazos(tiene)
  print(f"\n##### {ds} | bloque {BLOQUE.get(ds, '?')} | "
        f"{'con' if tiene else 'sin'} grupos one-hot | {len(lista)} brazos #####", flush=True)
  for emb in a.embedders:
    for ipc in a.ipcs:
      for mode, gamma in lista:
        etq = mode if mode != "penalty" else f"penalty@{gamma:g}"
        print(f"\n===== {ds} | {emb} | ipc {ipc} | {etq} | {a.runs} corridas =====", flush=True)
        for run in range(a.runs):
            if (ds, emb, ipc, etq, run) in hecho:
                continue
            seed = 132 + run
            set_seed(seed)
            data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
            P = OneHotProjector(data["X_train"])
            cfg = {"device": DEV, "ipc": ipc, "dm_iters": a.iters, "dm_lr": 0.5,
                   "dm_batch_real": 128, "dm_embedder_type": emb,
                   "dm_embedder_size": "base", "dm_embed_hidden": 256,
                   "dm_embed_dim": max(4, min(48, ipc - 2)),
                   "random_seed": seed, "init_seed": seed,
                   # -1 = initialisation (real rows), iters = the last iterate
                   "snapshot_every": a.iters, "return_snapshots": True,
                   "dm_onehot_mode": mode,
                   "dm_onehot_gamma": (gamma if mode == "penalty" else 100.0),
                   "dm_onehot_tau0": a.tau0, "dm_onehot_tau1": a.tau1}
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
            if mode in DISCRETOS and P:
                assert torch.equal(P.hard(X_syn), X_syn), f"{mode} devolvio filas no legales"

            cands = [("bestloss", X_syn), ("final", snap[a.iters]), ("init", snap[-1])]
            # Raw and projected, so "did the constraint help the optimisation" stays
            # separable from "did the returned rows happen to be legal". On the
            # discrete modes the projected column is the same number by the assert
            # above, which is what makes it a free consistency check.
            vistas = {"crudo": lambda X: X}
            if P:
                vistas["proyectado"] = P.hard
            t1 = time.time()
            for vista, fn in vistas.items():
                proyectados = [(tag, fn(Xc)) for tag, Xc in cands]
                for clf in CLFS:
                    accs = {tag: fit_eval(Xc, y_syn, data, clf, cfg, seed)
                            for tag, Xc in proyectados}
                    rows.append(dict(
                        dataset=ds, bloque=BLOQUE.get(ds, "?"),
                        tipo=("binario" if data["num_classes"] == 2 else "multiclase"),
                        num_classes=data["num_classes"], input_dim=data["input_dim"],
                        n_grupos=P.n_groups, n_dummy=P.n_dummy,
                        ipc=ipc, embedder=emb, mode=mode, gamma=gamma, brazo=etq,
                        tau0=a.tau0, tau1=a.tau1, eval_proj=vista,
                        run=run, seed=seed, classifier=clf,
                        acc_bestloss=accs["bestloss"][0], auc_bestloss=accs["bestloss"][1],
                        acc_final=accs["final"][0], auc_final=accs["final"][1],
                        acc_init=accs["init"][0], auc_init=accs["init"][1],
                        drift=drift, frac_fuera=fuera,
                        drift_final=drift_fin, frac_fuera_final=fuera_fin,
                        best_en_init=best_en_init,
                        synth_s=round(dt, 1), eval_s=round(time.time() - t1, 1)))
            hechos = rows[-len(CLFS):]
            print(f"[{ds:10s} {emb:10s} ipc{ipc:3d} {etq:12s} run{run:02d}] {dt:5.1f}s "
                  f"drift {drift:.3f}/{drift_fin:.3f}{' INIT' if best_en_init else ''} | " + "  ".join(
                f"{r['classifier']}={r['acc_bestloss']:.4f}" for r in hechos), flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False)

df = pd.DataFrame(rows)
df.to_csv(csv, index=False)
pd.set_option("display.width", 240)

print("\n=== accuracy media por brazo, conjunto de menor perdida, sin proyectar al evaluar ===")
c = df[df.eval_proj == "crudo"]
print(c.pivot_table(index=["bloque", "classifier"], columns="brazo",
                    values="acc_bestloss").round(4).to_string())

print("\n=== delta pareado contra none, por bloque y clasificador ===")
base = c[c.brazo == "none"].set_index(["dataset", "embedder", "ipc", "run", "classifier"]).acc_bestloss
filas = []
for brazo in sorted(c.brazo.unique()):
    if brazo == "none":
        continue
    s = c[c.brazo == brazo].set_index(["dataset", "embedder", "ipc", "run", "classifier"])
    d = (s.acc_bestloss - base).dropna()
    for (bl, clf), v in d.groupby([s.loc[d.index, "bloque"], d.index.get_level_values("classifier")]):
        filas.append(dict(brazo=brazo, bloque=bl, clasificador=clf, n=len(v),
                          media=v.mean(), mediana=v.median(), gana=int((v > 0.002).sum())))
print(pd.DataFrame(filas).round(4).to_string(index=False))

print("\n=== deriva de lo devuelto, por brazo (0 = sigue siendo one-hot) ===")
print(df.groupby("brazo")[["drift", "frac_fuera"]].mean().round(4).to_string())

# Only the datasets that have groups: on the others there is no projected view,
# and averaging a column over more cells than the one next to it would compare
# two different things.
print("\n=== dentro del bucle contra proyectar al evaluar, solo arboles y datasets con one-hot ===")
t = df[df.classifier.isin(["rf", "xgboost"]) & (df.n_grupos > 0)]
print(t.pivot_table(index="brazo", columns="eval_proj", values="acc_bestloss").round(4).to_string())

print(f"\nCSV -> {csv}")
