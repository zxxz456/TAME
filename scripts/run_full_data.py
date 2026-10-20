"""Accuracy and AUC of each classifier trained on the full training set.

Relative loss and relative regret (the paper's Eq. 14) need A_full, the score
of a classifier trained on all the training rows. The distillation sweeps only
store the distilled set (`acc_bestloss`) and the random real rows it started
from (`acc_init`), so this fills the missing third term. Nothing is distilled.

The split depends on the seed (`prepare_db` with `random_seed=seed`), so A_full
is measured per dataset, seed and classifier, on the same split, the same
classifier settings (`w1_common.fit_eval`) and the same seeds (132-141) as the
sweeps it is paired with.

    THREADS=4 bash scripts/run_sharded.sh run_full_data.py ~/tame_runs/full_data 4 \\
      magic spambase credit ... airlines
"""
import os, sys, time, argparse, contextlib, io
import pandas as pd
sys.path.insert(0, "/home/zxxz6/TAME"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/zxxz6/TAME")
from w1_common import BIN_SIN, MULTI_SIN, MIXTOS, SOLO_CAT, BLOQUE, CLFS, DEV, set_seed, fit_eval, reanudar
from data.prepare_database import prepare_db

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=BIN_SIN + MULTI_SIN + MIXTOS + SOLO_CAT)
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/full_data"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

csv = os.path.join(a.out, "results.csv")
rows, hecho = reanudar(csv, ["dataset", "seed"])
for ds in a.datasets:
    for run in range(a.runs):
        seed = 132 + run
        if (ds, seed) in hecho:
            continue
        set_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
        cfg = {"device": DEV, "random_seed": seed}
        for clf in CLFS:
            t0 = time.time()
            val, acc, auc = fit_eval(data["X_train"], data["y_train"], data, clf, cfg, seed)
            rows.append(dict(dataset=ds, bloque=BLOQUE.get(ds, "?"), seed=seed, run=run,
                             classifier=clf, n_train=int(data["X_train"].shape[0]),
                             val_full=val, acc_full=acc, auc_full=auc,
                             fit_s=round(time.time() - t0, 1)))
            print(f"[{ds:22s} s{seed}] {clf:8s} acc {acc:.4f} auc {auc:.4f} "
                  f"n={rows[-1]['n_train']} {rows[-1]['fit_s']}s", flush=True)
        pd.DataFrame(rows).to_csv(csv, index=False)

pd.DataFrame(rows).to_csv(csv, index=False)
print(f"CSV -> {csv}")
