"""Why plain TAME falls below random real rows on trees at high IPC.

On the datasets with no categoricals, TAME beats random rows on trees at IPC
10, and at IPC 200 it falls below them on most multi-class datasets (letter
-2.9, segment -2.6, pageblocks -1.4, pendigits -1.1 in the one-hot ablation)
and on two binary ones (spambase and madelon, -0.8). The W1 marginal term does
not close it, and on those datasets plain TAME does not damage the marginals.
This script tests no fix. It records what happens along the run, so that the
cause can be read off instead of guessed.

Five candidate causes, each with the signature it would leave:

  A  moving is harmful in itself: tree accuracy on validation falls steadily
     from the initialisation as the points move away from it
     (`desplazamiento`), rather than dropping once and settling
  B  the class boundaries are lost: matching moments pulls every point
     towards what is typical of its class, so the points move away from the
     other classes. `pureza` (share of the 10 nearest real neighbours that
     share the point's class) and `margen` rise above the init's, and
     `frontera` (share of points with at least one neighbour of another
     class) falls
  C  the joint structure is lost: the marginals stay put, but the rank
     correlation between pairs of columns drifts from the real one
     (`spearman`, mean absolute difference per class against the training
     class)
  D  it is the regime, not the task: at IPC 200 random rows are 87% of each
     class on segment and 37% on letter, against under 4% on credit, so the
     baseline is nearly the full data and anything that moves the rows can
     only lose information. This is read across datasets, not along the run:
     spambase (binary, 13% of each class, also loses at IPC 200) against
     credit (binary, 2.8%, TAME wins by 6.9)
  E  the ties are broken: many columns are integers (letter, pendigits, part
     of segment) and real rows repeat values. One step of SGD makes every
     value distinct, and a tree can then split between points that were
     equal, on noise. In a smoke run the trees dropped about 2 points after
     the FIRST step, with the points moved by 0.002. Signature: `en_rejilla`
     (share of entries that are a value observed in that training column)
     falls from 1 to 0 at once and the trees fall with it. And a direct test:
     every value is snapped to the nearest value observed in its column
     (`*_snap`, on the returned set and the last iterate); if the trees come
     back, this is it, and the fix is that projection, the continuous
     counterpart of the one-hot one

Every quantity is measured on the initialisation too, which is real rows of the
same size, so it is the reference for what sampling alone gives. The neighbour
measures use the validation split as the real reference, because the init rows
are training rows and would find themselves at distance zero.

Seeds 2000-2004, disjoint from the tuning (1000-1009) and the comparisons
(132-141). Snapshots every 50 iterations; each one is fitted with rf and xgboost
and read on validation and test. The sets themselves are kept, for whatever the
next question is.

    THREADS=4 bash scripts/run_sharded.sh run_multi_diag.py ~/tame_runs/multi_diag 4 \\
      letter segment spambase credit
    .venv/bin/python scripts/analyze_multi_diag.py ~/tame_runs/multi_diag/results.csv
"""
import os, sys, time, argparse, contextlib, io, warnings
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, "/home/zxxz6/TAME"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/zxxz6/TAME")
from w1_common import DEV, prepare_db, synthesize, set_seed, columnas_w1, marginales, fit_eval, parse_brazo

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["letter", "segment", "spambase", "credit"])
ap.add_argument("--arms", nargs="+", default=["none"], help="arm labels, as in w1_common")
ap.add_argument("--ipcs", nargs="+", type=int, default=[200])
ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2000, 2005)))
ap.add_argument("--embedders", nargs="+", default=["ln_res_l"])
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--every", type=int, default=50, help="snapshot spacing")
ap.add_argument("--k", type=int, default=10, help="neighbours for pureza and frontera")
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/multi_diag"))
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "conjuntos"), exist_ok=True)
ARB = ["rf", "xgboost"]


def spearman_clase(X, C, ipc=None, y=None):
    """Rank correlation matrix per class: (C, k, k). Constant columns give NaN."""
    out = []
    for c in range(C):
        Xc = X[c * ipc:(c + 1) * ipc] if y is None else X[y == c]
        # A column constant within the class has no rank correlation: NaN, and
        # left out of the mean
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.append(np.atleast_2d(stats.spearmanr(Xc).statistic))
    return np.stack(out)


class Medidor:
    """Everything that is measured on a set, with the references computed once
    per distillation."""

    def __init__(self, data, cols, ipc, k):
        self.C, self.ipc, self.cols, self.data = data["num_classes"], ipc, cols, data
        Xv = data["X_val"].cpu().numpy()
        self.yv = data["y_val"].cpu().numpy().astype(int)
        self.nn = NearestNeighbors(n_neighbors=k).fit(Xv)
        # nearest real point of the same class and of any other class
        self.nn_c = [NearestNeighbors(n_neighbors=1).fit(Xv[self.yv == c]) for c in range(self.C)]
        self.nn_no = [NearestNeighbors(n_neighbors=1).fit(Xv[self.yv != c]) for c in range(self.C)]
        Xt = data["X_train"].cpu().numpy()[:, cols]
        yt = data["y_train"].cpu().numpy().astype(int)
        self.sp_real = spearman_clase(Xt, self.C, y=yt)
        self.iu = np.triu_indices(len(cols), 1)
        # every value observed in each training column, for E
        self.rejilla = {j: np.unique(data["X_train"][:, j].cpu().numpy()) for j in cols.tolist()}

    def ajustar(self, X):
        """Each value to the nearest value observed in its training column."""
        X = X.detach().cpu().numpy().copy()
        for j, g in self.rejilla.items():
            i = np.clip(np.searchsorted(g, X[:, j]), 1, len(g) - 1)
            izq, der = g[i - 1], g[i]
            X[:, j] = np.where(np.abs(X[:, j] - izq) <= np.abs(der - X[:, j]), izq, der)
        return torch.as_tensor(X)

    def medir(self, X, X0):
        X = X.detach().cpu().numpy(); X0 = X0.detach().cpu().numpy()
        y = np.repeat(np.arange(self.C), self.ipc)
        _, idx = self.nn.kneighbors(X)
        pureza = (self.yv[idx] == y[:, None]).mean(1)
        margen = []
        for c in range(self.C):
            Xc = X[y == c]
            dp = self.nn_c[c].kneighbors(Xc)[0][:, 0]
            do = self.nn_no[c].kneighbors(Xc)[0][:, 0]
            margen.append((do - dp) / (do + dp + 1e-12))
        margen = np.concatenate(margen)
        div = np.mean([np.mean(np.linalg.norm(X[y == c][:, None] - X[y == c][None], axis=-1))
                       for c in range(self.C)])
        sp = spearman_clase(X[:, self.cols], self.C, ipc=self.ipc)
        d_sp = np.nanmean(np.abs(sp - self.sp_real)[:, self.iu[0], self.iu[1]])
        m = marginales(torch.as_tensor(X), self.data, self.cols, self.ipc)
        en_rej = np.mean([np.isin(X[:, j], g).mean() for j, g in self.rejilla.items()])
        distintos = np.mean([len(np.unique(X[:, j])) / len(X) for j in self.rejilla])
        return dict(pureza=pureza.mean(), frontera=(pureza < 1).mean(), margen=margen.mean(),
                    diversidad=div, spearman=d_sp, w1=m["w1"], sw=m["sw"],
                    en_rejilla=en_rej, distintos=distintos,
                    desplazamiento=np.linalg.norm(X - X0, axis=1).mean() / np.sqrt(X.shape[1]))


csv = os.path.join(a.out, "results.csv")
prev = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
rows = prev.to_dict("records")
hecho = (set(map(tuple, prev[["dataset", "embedder", "ipc", "brazo", "seed"]].drop_duplicates().values))
         if len(prev) else set())
for ds in a.datasets:
  for emb in a.embedders:
    for ipc in a.ipcs:
      for etq in a.arms:
        for seed in a.seeds:
            if (ds, emb, ipc, etq, seed) in hecho:
                continue
            modo, peso = parse_brazo(etq)
            set_seed(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
            cols = columnas_w1(data["X_train"])
            n_cls = np.bincount(data["y_train"].cpu().numpy().astype(int))
            cfg = {"device": DEV, "ipc": ipc, "dm_iters": a.iters, "dm_lr": 0.5,
                   "dm_batch_real": 128, "dm_embedder_type": emb,
                   "dm_embedder_size": "base", "dm_embed_hidden": 256,
                   "dm_embed_dim": max(4, min(48, ipc - 2)),
                   "random_seed": seed, "init_seed": seed,
                   "snapshot_every": a.every, "return_snapshots": True,
                   "dm_onehot_mode": modo, "dm_w1_weight": peso}
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                X_syn, y_syn, snaps = synthesize("tame", data, cfg)
            dt = time.time() - t0
            snap = dict(snaps)
            torch.save({"snapshots": {i: x.cpu() for i, x in snap.items()}, "best": X_syn.cpu(),
                        "y": y_syn.cpu(), "cfg": {k: v for k, v in cfg.items() if k != "device"}},
                       os.path.join(a.out, "conjuntos", f"{ds}_{emb}_ipc{ipc}_{etq}_s{seed}.pt"))
            med = Medidor(data, cols, ipc, a.k)
            X0 = snap[-1]
            t1 = time.time()
            nuevas = []
            for it, X in [("best", X_syn)] + sorted(snap.items()):
                fila = dict(dataset=ds, embedder=emb, ipc=ipc, brazo=etq, seed=seed,
                            num_classes=data["num_classes"],
                            cobertura=float(np.mean(np.minimum(ipc / n_cls, 1))),
                            iteracion=(-2 if it == "best" else it), es_best=(it == "best"),
                            **med.medir(X, X0))
                for clf in ARB:
                    fila[f"val_{clf}"], fila[f"acc_{clf}"], _ = fit_eval(X, y_syn, data, clf, cfg, seed)
                # E's direct test, on the two sets that matter
                if it in ("best", a.iters):
                    Xs = med.ajustar(X)
                    for clf in ARB:
                        fila[f"val_{clf}_snap"], fila[f"acc_{clf}_snap"], _ = fit_eval(
                            Xs, y_syn, data, clf, cfg, seed)
                nuevas.append(fila)
            for r in nuevas:
                r.update(synth_s=round(dt, 1), eval_s=round(time.time() - t1, 1))
            rows += nuevas
            r0 = {r["iteracion"]: r for r in nuevas}
            ini, fin = r0[-1], r0[a.iters]
            print(f"[{ds:10s} {emb:10s} ipc{ipc:3d} {etq:10s} s{seed}] {dt:5.1f}s+{time.time() - t1:4.0f}s | "
                  f"arboles val {np.mean([ini['val_rf'], ini['val_xgboost']]):.3f}>"
                  f"{np.mean([fin['val_rf'], fin['val_xgboost']]):.3f} | pureza {ini['pureza']:.3f}>{fin['pureza']:.3f} "
                  f"rejilla {ini['en_rejilla']:.2f}>{fin['en_rejilla']:.2f} "
                  f"snap {np.mean([fin['val_rf_snap'], fin['val_xgboost_snap']]):.3f} | "
                  f"spearman {ini['spearman']:.3f}>{fin['spearman']:.3f} desplaz {fin['desplazamiento']:.3f}",
                  flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False)

print(f"\nCSV -> {csv}")
