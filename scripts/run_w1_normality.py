"""Shapiro-Wilk and Mardia of every distilled set, plain TAME against W1D.

Section 5.3 of the paper measures how Gaussian the distilled data is with two
statistics (Table 14): the Shapiro-Wilk pass rate, per column, and Mardia's
multivariate kurtosis, per class. Phase 2 of the W1 experiment stored only a
pooled Shapiro-Wilk and did not keep the sets, so this re-runs the same
distillations (10 datasets, IPC 10/50/200, both embedders, seeds 132-141,
`none` and `w1@<weight>`) and measures both statistics, the paper's way, on
three sets per distillation: the initialisation (real rows, the reference of
the same size), the returned set and the last iterate. No classifier is
trained, which is 86% of the time of a normal sweep; accuracy comes from the
phase 2 results, which ran the same design.

Per set, per class, then averaged over classes (scripts/run_normality_test.py,
the code behind Table 14):

  sw_pass      share of columns whose Shapiro-Wilk p > 0.05; higher = more
               Gaussian. Constant columns are skipped
  sw_w         mean Shapiro-Wilk W over the columns, same direction, less coarse
  mardia_dev   |b2,p - p(p+2)|, Mardia's kurtosis against its Gaussian value;
               lower = more Gaussian. This is Table 14's column
  mardia_rel   the same divided by p(p+2), so datasets with different p compare
  mardia_skew_pass, mardia_kurt_pass  share of classes passing each test

Mardia needs at least p + 10 rows per class (`mardia_test` returns nothing
otherwise), so it is missing at IPC 10 on every dataset, and at IPC 50 on
spambase (p = 57) and madelon (p = 500). It is a sample-size statistic: compare
a distilled set with its own initialisation, never with the full data.

`real.csv` holds the same statistics on the full training set per dataset (n
capped as in the paper), for reference only, since n differs.

    bash scripts/run_sharded.sh run_w1_normality.py ~/tame_runs/w1_normality 10 \\
      letter higgs pendigits satimage segment credit magic spambase madelon pageblocks
"""
import os, sys, time, argparse, contextlib, io, warnings
import numpy as np, pandas as pd, torch
from scipy import stats
sys.path.insert(0, "/home/zxxz6/TAME"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/zxxz6/TAME")
from w1_common import BIN_SIN, MULTI_SIN, BLOQUE, DEV, prepare_db, synthesize, set_seed, parse_brazo
from run_normality_test import mardia_test

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=BIN_SIN + MULTI_SIN)
ap.add_argument("--weight", type=float, default=0.03, help="the W1 weight, as in phase 2")
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--embedders", nargs="+", default=["ln_res_l", "dcnv2_base"])
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/w1_normality"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
BRAZOS = ["none", f"w1@{a.weight:g}"]


def normalidad(X, y, seed=0):
    """The paper's statistics on one set, per class and then averaged."""
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
    passes, ws, devs, rels, sk, ku = [], [], [], [], [], []
    rng_state = np.random.get_state(); np.random.seed(seed)   # mardia_test subsamples
    for c in np.unique(y):
        Xc = X[y == c]
        if len(Xc) < 10:
            continue
        p_c, w_c = [], []
        for j in range(Xc.shape[1]):
            col = Xc[:, j]
            if np.std(col) < 1e-10:
                continue
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
            sk.append(m["skewness_p"] > 0.05); ku.append(m["kurtosis_p"] > 0.05)
    np.random.set_state(rng_state)
    f = lambda v: float(np.mean(v)) if len(v) else np.nan
    return dict(sw_pass=f(passes), sw_w=f(ws), mardia_dev=f(devs), mardia_rel=f(rels),
                mardia_skew_pass=f(sk), mardia_kurt_pass=f(ku), mardia_clases=len(devs))


csv = os.path.join(a.out, "results.csv")
prev = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
rows = prev.to_dict("records")
hecho = (set(map(tuple, prev[["dataset", "embedder", "ipc", "brazo", "seed"]].drop_duplicates().values))
         if len(prev) else set())
real_csv = os.path.join(a.out, "real.csv")
reales = pd.read_csv(real_csv).to_dict("records") if os.path.exists(real_csv) else []

for ds in a.datasets:
    if not any(r["dataset"] == ds for r in reales):
        with contextlib.redirect_stdout(io.StringIO()):
            data = prepare_db({"random_seed": 132, "device": "cpu"}, name=ds)
        reales.append(dict(dataset=ds, bloque=BLOQUE.get(ds, "?"), input_dim=data["input_dim"],
                           n=len(data["y_train"]),
                           **normalidad(data["X_train"].numpy(), data["y_train"].numpy())))
        pd.DataFrame(reales).to_csv(real_csv, index=False)
    print(f"##### {ds} | {BRAZOS}", flush=True)
    for emb in a.embedders:
        for ipc in a.ipcs:
            for etq in BRAZOS:
                for run in range(a.runs):
                    seed = 132 + run
                    if (ds, emb, ipc, etq, seed) in hecho:
                        continue
                    modo, peso = parse_brazo(etq)
                    set_seed(seed)
                    with contextlib.redirect_stdout(io.StringIO()):
                        data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
                    cfg = {"device": DEV, "ipc": ipc, "dm_iters": a.iters, "dm_lr": 0.5,
                           "dm_batch_real": 128, "dm_embedder_type": emb,
                           "dm_embedder_size": "base", "dm_embed_hidden": 256,
                           "dm_embed_dim": max(4, min(48, ipc - 2)),
                           "random_seed": seed, "init_seed": seed,
                           "snapshot_every": a.iters, "return_snapshots": True,
                           "dm_onehot_mode": modo, "dm_w1_weight": peso}
                    t0 = time.time()
                    with contextlib.redirect_stdout(io.StringIO()):
                        X_syn, y_syn, snaps = synthesize("tame", data, cfg)
                    dt = time.time() - t0
                    snap = dict(snaps)
                    y = y_syn.cpu().numpy()
                    nuevas = []
                    for conjunto, X in (("init", snap[-1]), ("devuelto", X_syn), ("ultimo", snap[a.iters])):
                        nuevas.append(dict(dataset=ds, bloque=BLOQUE.get(ds, "?"),
                                           num_classes=data["num_classes"], input_dim=data["input_dim"],
                                           ipc=ipc, embedder=emb, brazo=etq, w1_weight=peso,
                                           run=run, seed=seed, conjunto=conjunto, synth_s=round(dt, 1),
                                           **normalidad(X.cpu().numpy(), y, seed)))
                    rows += nuevas
                    i, b = nuevas[0], nuevas[1]
                    print(f"[{ds:10s} {emb:10s} ipc{ipc:3d} {etq:9s} s{seed}] {dt:5.1f}s | "
                          f"sw_pass {i['sw_pass']:.3f}>{b['sw_pass']:.3f} sw_w {i['sw_w']:.3f}>{b['sw_w']:.3f} | "
                          f"mardia_rel {i['mardia_rel']:.3f}>{b['mardia_rel']:.3f}", flush=True)
                    pd.DataFrame(rows).to_csv(csv, index=False)

df = pd.DataFrame(rows)
df.to_csv(csv, index=False)
pd.set_option("display.width", 240)
print(df[df.conjunto != "ultimo"].pivot_table(index=["bloque", "ipc"], columns=["brazo", "conjunto"],
                                              values=["sw_pass", "mardia_rel"]).round(3).to_string())
print(f"\nCSV -> {csv}")
