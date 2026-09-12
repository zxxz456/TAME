"""TAME vs TAME-critic, both with snapshots, scored under best-loss and best-val.

Table 11 could only compare the base versions, because the critic variant kept no
snapshot trail and so could not take part in the validation-based selection of
Sec. 4.2. With the trail in place both arms can be scored under the same rule.

Selection uses the validation split only; test is read once, after the choice is
frozen.
"""
import os, sys, time, random, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/zxxz6/TAME"); os.chdir("/home/zxxz6/TAME")
from data.prepare_database import prepare_db
from synth.registry import synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="adult")
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--ipc", type=int, default=50)
ap.add_argument("--snapshot-every", type=int, default=100)
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/critic_vs_tame"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLFS = ["mlp", "rf", "xgboost"]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def fit(X, y, data, clf, cfg, seed):
    set_seed(seed)
    td = {"X_train": X, "y_train": y, "X_val": data["X_val"], "y_val": data["y_val"],
          "input_dim": data["input_dim"], "num_classes": data["num_classes"]}
    return train_classifier(td, {**cfg, "classifier": clf,
                                 "classifier_hidden": [128, 64], "classifier_epochs": 20})

def val_acc(model, data):
    v = dict(data); v["X_test"], v["y_test"] = data["X_val"], data["y_val"]
    return float(evaluate_classifier(model, v, DEV)[0])

rows = []
for run in range(a.runs):
    seed = 132 + run
    for arm, use_critic in [("tame", False), ("critic", True)]:
        set_seed(seed)
        data = prepare_db({"random_seed": seed, "device": DEV}, name=a.dataset)
        cfg = {
            "device": DEV, "ipc": a.ipc, "dm_iters": a.iters, "dm_lr": 0.5,
            "dm_batch_real": 128, "dm_embedder_type": "ln_res_l",
            "dm_embedder_size": "base", "dm_embed_hidden": 256,
            "dm_embed_dim": max(4, min(48, a.ipc - 2)),
            "snapshot_every": a.snapshot_every, "return_snapshots": True,
            "random_seed": seed,
        }
        if use_critic:
            cfg.update(dm_use_critic=True, dm_adv_weight=0.05, dm_n_critic=3,
                       dm_critic_clean_selection=True)
        stype = "tame_critic" if use_critic else "tame"

        t0 = time.time()
        X_best, y_syn, snaps = synthesize(stype, data, cfg)
        dt = time.time() - t0

        for clf in CLFS:
            # best-loss: the set the synthesizer returns
            m = fit(X_best, y_syn, data, clf, cfg, seed)
            acc_bl = float(evaluate_classifier(m, data, DEV)[0])

            # best-val: pick among the trail by validation accuracy, then read test once
            best_v, best_m, best_i = -1.0, None, None
            for it, Xc in snaps:
                mc = fit(Xc, y_syn, data, clf, cfg, seed)
                v = val_acc(mc, data)
                if v > best_v:
                    best_v, best_m, best_i = v, mc, it
            acc_bv = float(evaluate_classifier(best_m, data, DEV)[0])

            rows.append(dict(dataset=a.dataset, arm=arm, run=run, seed=seed,
                             classifier=clf, acc_bestloss=acc_bl, acc_bestval=acc_bv,
                             sel_iter=best_i, n_cand=len(snaps), synth_s=round(dt, 1)))
        print(f"[{arm:6s} run{run:02d}] {dt:5.1f}s  " + "  ".join(
            f"{r['classifier']}: bl={r['acc_bestloss']:.4f} bv={r['acc_bestval']:.4f}@{r['sel_iter']}"
            for r in rows[-3:]), flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "results.csv"), index=False)
piv = df.groupby(["arm", "classifier"])[["acc_bestloss", "acc_bestval"]].agg(["mean", "std"])
print("\n=== accuracy ===")
print(piv.round(4))
print("\n=== ganancia de best-val sobre best-loss ===")
g = df.assign(gain=df.acc_bestval - df.acc_bestloss)
print(g.groupby(["arm", "classifier"]).gain.agg(["mean", "std"]).round(4))
print("\n=== iteracion seleccionada ===")
print(df.groupby(["arm", "classifier"]).sel_iter.agg(["mean", "median"]).round(1))
print("\nCSV ->", os.path.join(a.out, "results.csv"))
