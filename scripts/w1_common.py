"""What the marginal-term experiments share: datasets, arms, one distillation and
its evaluation.

`run_w1_tuning.py` picks the weight of the W1 term on validation and
`run_w1_vs_tame.py` compares the chosen weight against plain TAME on test. Both
run the same distillation and read it the same way, so that part lives here
once instead of twice.

An arm is a label that fixes the two switches of the loop:

  none          plain TAME, the published loop
  w1@L          plain TAME plus the W1 term with weight L
  ste           the in-loop one-hot projection, for the datasets with
                categoricals later on
  ste+w1@L      both, each on its own columns

Every distilled set is read three ways: the lowest-loss set the loop returns,
the last iterate and the initialisation, which is real rows. For each one the
row holds validation and test accuracy, test AUC, and two measures of the
marginals that say whether the term did what it is for:

  w1   exact 1-D Wasserstein distance between each synthetic column and the
       same column of the real training class, averaged over columns and
       classes. Lower is closer to the real marginals.
  sw   Shapiro-Wilk W per column, over all the rows of the set together,
       averaged over columns; swp is the share of columns with p > 0.05, the
       paper's pass rate (Table 14). Higher is more Gaussian. The rows are
       pooled because a tree split sees the whole column, and the init is the
       reference because it has the same number of rows: normality tests depend
       on the sample size, so only same-size comparisons mean anything.

Both are measured only on the columns the W1 term acts on, the ones outside the
one-hot groups.
"""
import os, sys, time, random, contextlib, io, warnings
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.datasets import fetch_openml
sys.path.insert(0, "/home/zxxz6/TAME")
from data.prepare_database import prepare_db, DATASET_REGISTRY
from data.tdbench_datasets import register_tdbench_datasets, _preprocess_and_split
from synth.registry import synthesize
from synth.onehot import OneHotProjector
from synth.snap import GridSnapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_normality_test import mardia_test
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLFS = ["mlp", "rf", "xgboost"]

# The two null blocks of the one-hot ablation: no one-hot groups, so every
# column is continuous to the loop and the W1 term acts on all of them.
BIN_SIN = ["magic", "spambase", "credit", "madelon", "higgs"]
MULTI_SIN = ["letter", "pendigits", "satimage", "segment", "pageblocks"]
# The datasets with both kinds of column, for the ste x W1 design that follows
# if the term works. Not run by default.
MIXTOS = ["adult", "bank", "german", "law_school_admissions", "covertype",
          # the other datasets of the registry with both one-hot groups and continuous
          # columns, all binary; covertype is the only multi-class one
          "electricity", "airline_satisfaction", "road_safety", "medical_appointments",
          "hcdr", "airlines"]
BLOQUE = {**{d: "binario-sin" for d in BIN_SIN}, **{d: "multi-sin" for d in MULTI_SIN},
          **{d: "mixto-binario" for d in MIXTOS if d != "covertype"}, "covertype": "mixto-multi"}

_BAJADOS = {}


def _bajar(nombre, version, intentos=6, espera=10):
    """fetch_openml once per process, retried on network errors."""
    k = (nombre, version)
    if k not in _BAJADOS:
        for i in range(intentos):
            try:
                _BAJADOS[k] = fetch_openml(nombre, version=version, as_frame=True)
                break
            except OSError as e:
                if i == intentos - 1:
                    raise
                print(f"{nombre}: network error ({e}); retrying in {espera * 2 ** i}s", flush=True)
                time.sleep(espera * 2 ** i)
    return _BAJADOS[k]


def _covertype(random_seed=42, device="cpu", max_rows=100_000):
    """covertype from OpenML, subsampled as in the one-hot experiments."""
    ds = _bajar("covertype", 3)
    X_df, y = ds.data, pd.Series(np.asarray(ds.target).ravel()).astype(str)
    idx = np.random.default_rng(0).choice(len(y), max_rows, replace=False)
    return _preprocess_and_split(X_df.iloc[idx].reset_index(drop=True),
                                 y.iloc[idx].reset_index(drop=True), random_seed, device)


register_tdbench_datasets()
DATASET_REGISTRY.setdefault("covertype", _covertype)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def parse_brazo(etq):
    """Arm label to (onehot_mode, w1_weight). A `snap` part switches on the
    in-loop snap to observed values; read it with `usa_snap`."""
    modo, peso = "none", 0.0
    for parte in etq.split("+"):
        if parte.startswith("w1@"):
            peso = float(parte[3:])
        elif parte in ("none", "ste"):
            modo = parte
        elif parte != "snap":
            raise ValueError(f"unknown arm {etq!r}")
    return modo, peso


def usa_snap(etq):
    """Whether the arm snaps to observed values inside the loop (synth/snap.py)."""
    return "snap" in etq.split("+")


def columnas_w1(X_train):
    """The columns outside the one-hot groups, the same ones the loop uses."""
    en_grupo = torch.zeros(X_train.shape[1], dtype=torch.bool)
    for g, _, _ in OneHotProjector(X_train).groups:
        en_grupo[g.cpu()] = True
    return torch.nonzero(~en_grupo).squeeze(1).numpy()


def marginales(X, data, cols, ipc):
    """w1, sw and swp of one set, on the columns `cols` (see the module doc)."""
    if len(cols) == 0:
        return dict(w1=np.nan, sw=np.nan, swp=np.nan)
    X = X.detach().cpu().numpy()[:, cols]
    Xr = data["X_train"].cpu().numpy()[:, cols]
    yr = data["y_train"].cpu().numpy()
    C = data["num_classes"]
    w1 = np.mean([stats.wasserstein_distance(X[c * ipc:(c + 1) * ipc, j], Xr[yr == c, j])
                  for c in range(C) for j in range(len(cols))])
    ws, ps = [], []
    for j in range(len(cols)):
        col = X[:, j]
        if np.ptp(col) < 1e-10:
            continue                      # constant: SW returns W=1 and inflates the rate
        r = stats.shapiro(col)
        ws.append(r.statistic); ps.append(r.pvalue)
    return dict(w1=float(w1), sw=float(np.mean(ws)) if ws else np.nan,
                swp=float(np.mean(np.array(ps) > 0.05)) if ps else np.nan)


def normalidad(X, y, seed=0):
    """The paper's normality statistics on one set (Table 14, run_normality_test.py):
    per class, then averaged over classes.

      swc_pass    share of columns with Shapiro-Wilk p > 0.05 (higher = more Gaussian)
      swc_w       mean Shapiro-Wilk W over columns, same direction, less coarse
      mardia_dev  |b2,p - p(p+2)|, Mardia's kurtosis against its Gaussian value
                  (lower = more Gaussian); NaN when a class has fewer than p + 10 rows
      mardia_rel  the same divided by p(p+2), comparable across datasets
    """
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
    passes, ws, devs, rels = [], [], [], []
    estado = np.random.get_state(); np.random.seed(seed)       # mardia_test subsamples
    for c in np.unique(y):
        Xc = X[y == c]
        if len(Xc) < 10:
            continue
        p_c, w_c = [], []
        for j in range(Xc.shape[1]):
            col = Xc[:, j]
            if np.std(col) < 1e-10:
                continue                      # constant: SW returns W=1 and inflates the rate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = stats.shapiro(col[:5000])
            p_c.append(r.pvalue); w_c.append(r.statistic)
        if p_c:
            passes.append(np.mean(np.array(p_c) > 0.05)); ws.append(np.mean(w_c))
        m = mardia_test(Xc)
        if m is not None:
            e = m["kurtosis_expected"]
            devs.append(abs(m["kurtosis_stat"] - e)); rels.append(abs(m["kurtosis_stat"] - e) / e)
    np.random.set_state(estado)
    f = lambda v: float(np.mean(v)) if len(v) else np.nan
    return dict(swc_pass=f(passes), swc_w=f(ws), mardia_dev=f(devs), mardia_rel=f(rels))


def fit_eval(X_syn, y_syn, data, clf, cfg, seed):
    """One classifier, read on validation and on test: (val acc, test acc, test AUC)."""
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


def correr(ds, emb, ipc, etq, seed, iters=1000, extra=None, posthoc_snap=False,
           medir_normalidad=False):
    """One distillation and its evaluation: one row per classifier.

    The split, the initialisation and the classifiers are all seeded with
    `seed`, so two arms with the same seed start from the same real rows and
    differ only in the loop. The embedder draws inside the loop are seeded from
    the clock (models/embedders.py), so even two identical arms do not produce
    the same trajectory

    With `posthoc_snap` the same sets are also read snapped to observed values
    after distillation, as two more arms: `<etq>+redondeo` (always snapped) and
    `<etq>+valid` (per set and classifier, snapped or not, whichever scores
    higher on validation). They cost classifier fits, not distillations

    With `medir_normalidad` every set also gets the paper's per-class
    Shapiro-Wilk and Mardia (`normalidad`)
    """
    modo, peso = parse_brazo(etq)
    set_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
    cols = columnas_w1(data["X_train"])
    P = OneHotProjector(data["X_train"])
    cfg = {"device": DEV, "ipc": ipc, "dm_iters": iters, "dm_lr": 0.5,
           "dm_batch_real": 128, "dm_embedder_type": emb,
           "dm_embedder_size": "base", "dm_embed_hidden": 256,
           "dm_embed_dim": max(4, min(48, ipc - 2)),
           "random_seed": seed, "init_seed": seed,
           # -1 = initialisation (real rows), iters = the last iterate
           "snapshot_every": iters, "return_snapshots": True,
           "dm_onehot_mode": modo, "dm_w1_weight": peso, "dm_snap": usa_snap(etq)}
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        X_syn, y_syn, snaps = synthesize("tame", data, cfg)
    dt = time.time() - t0
    snap = dict(snaps)
    cands = {"bestloss": X_syn, "final": snap[iters], "init": snap[-1]}
    G = GridSnapper(data["X_train"], torch.as_tensor(cols))
    vistas = {etq: cands}
    if posthoc_snap:
        vistas[f"{etq}+redondeo"] = {k: G.hard(X) for k, X in cands.items()}
    t1 = time.time()
    res, marg = {}, {}
    for arm, cs in vistas.items():
        marg[arm] = {k: marginales(X, data, cols, ipc) for k, X in cs.items()}
        if medir_normalidad:
            for k, X in cs.items():
                marg[arm][k].update(normalidad(X.detach().cpu().numpy(), y_syn.cpu().numpy(), seed))
        res[arm] = {clf: {k: fit_eval(X, y_syn, data, clf, cfg, seed) for k, X in cs.items()} for clf in CLFS}
        marg[arm]["rejilla"] = G.on_grid(cs["bestloss"])
    if posthoc_snap:
        # validation picks, per set and classifier, between the raw and the snapped read
        crudo, red = res[etq], res[f"{etq}+redondeo"]
        res[f"{etq}+valid"] = {clf: {k: (red[clf][k] if red[clf][k][0] > crudo[clf][k][0] else crudo[clf][k])
                                     for k in cands} for clf in CLFS}
        marg[f"{etq}+valid"] = marg[etq]
    rows = []
    for arm in res:
        for clf in CLFS:
            fila = dict(
                dataset=ds, bloque=BLOQUE.get(ds, "?"),
                tipo=("binario" if data["num_classes"] == 2 else "multiclase"),
                num_classes=data["num_classes"], input_dim=data["input_dim"],
                n_cont=len(cols), n_dummy=P.n_dummy,
                ipc=ipc, embedder=emb, brazo=arm, onehot_mode=modo, w1_weight=peso,
                snap_loop=usa_snap(etq), seed=seed, classifier=clf,
                best_en_init=bool(torch.equal(X_syn, snap[-1])),
                en_rejilla=marg[arm]["rejilla"])
            for k in cands:
                fila[f"val_{k}"], fila[f"acc_{k}"], fila[f"auc_{k}"] = res[arm][clf][k]
                for m in ("w1", "sw", "swp", "swc_pass", "swc_w", "mardia_dev", "mardia_rel"):
                    if m in marg[arm][k]:
                        fila[f"{m}_{k}"] = marg[arm][k][m]
            fila.update(synth_s=round(dt, 1), eval_s=round(time.time() - t1, 1))
            rows.append(fila)
    return rows


def linea(rows, dt_tag=""):
    """The log line of one distillation: what the monitors count."""
    r0 = rows[0]
    return (f"[{r0['dataset']:10s} {r0['embedder']:10s} ipc{r0['ipc']:3d} {r0['brazo']:12s} "
            f"s{r0['seed']}] {r0['synth_s']:5.1f}s "
            f"w1 {r0['w1_init']:.3f}>{r0['w1_bestloss']:.3f} sw {r0['sw_init']:.3f}>{r0['sw_bestloss']:.3f}"
            f"{' INIT' if r0['best_en_init'] else ''} | "
            + "  ".join(f"{x['classifier']}={x['val_bestloss']:.3f}/{x['acc_bestloss']:.3f}"
                        for x in rows))


def reanudar(csv, clave):
    """Rows already written and the set of distillations they cover."""
    prev = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    hecho = set(map(tuple, prev[clave].drop_duplicates().values)) if len(prev) else set()
    if hecho:
        print(f"resuming: {len(hecho)} distillations already in {csv}", flush=True)
    return prev.to_dict("records"), hecho
