"""NODE with zero-initialised tree parameters vs randomly initialised ones.

TAME never trains the embedder, so NODE's feature_logits and thresholds stay at
their zero init: softmax(zeros) is uniform, every tree and level reads the same
mean of all coordinates, and the ensemble contributes nothing (its singular
values are zero to three decimals, see scripts/check_node_degeneracy.py).

The paper reads NODE's poor score as evidence that emulating trees in the
embedder does not help. If the ensemble was inert, that conclusion never had a
tree-like embedder behind it. This runs both arms on the same seeds to find out.

Both arms are identical except for tree_random_init, so any difference is
attributable to it.
"""
import os, sys, time, random, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/zxxz6/TAME"); os.chdir("/home/zxxz6/TAME")
from data.prepare_database import prepare_db
from synth.registry import synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["adult"])
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--ipc", type=int, default=50)
ap.add_argument("--logit-std", type=float, default=3.0)
ap.add_argument("--thr-std", type=float, default=1.0)
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/node_random_init"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLFS = ["mlp", "rf", "xgboost"]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

rows = []
for ds in a.datasets:
    # Arm-major: every zero run first, then every random run. Seeds are shared
    # across arms, so the pairing survives the reordering.
    for arm, rnd in [("zero", False), ("random", True)]:
        print(f"\n===== {ds} | brazo {arm} | {a.runs} corridas =====", flush=True)
        for run in range(a.runs):
            seed = 132 + run
            set_seed(seed)
            data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
            cfg = {
                "device": DEV, "ipc": a.ipc, "dm_iters": a.iters, "dm_lr": 0.5,
                "dm_batch_real": 128, "dm_embedder_type": "node",
                "dm_embedder_size": "base", "dm_embed_hidden": 256,
                "dm_embed_dim": max(4, min(48, a.ipc - 2)),
                "dm_embedder_overrides": {
                    "tree_random_init": rnd,
                    "logit_std": a.logit_std, "thr_std": a.thr_std,
                },
                "random_seed": seed,
            }
            t0 = time.time()
            X_syn, y_syn = synthesize("tame", data, cfg)
            dt = time.time() - t0

            for clf in CLFS:
                set_seed(seed)
                td = {"X_train": X_syn, "y_train": y_syn,
                      "X_val": data["X_val"], "y_val": data["y_val"],
                      "input_dim": data["input_dim"], "num_classes": data["num_classes"]}
                m = train_classifier(td, {**cfg, "classifier": clf,
                                          "classifier_hidden": [128, 64],
                                          "classifier_epochs": 20})
                acc, auc = evaluate_classifier(m, data, DEV)
                rows.append(dict(dataset=ds, arm=arm, run=run, seed=seed,
                                 classifier=clf, acc=float(acc), auc=float(auc),
                                 synth_s=round(dt, 1)))
            print(f"[{ds:12s} {arm:6s} run{run:02d}] {dt:5.1f}s  " + "  ".join(
                f"{r['classifier']}={r['acc']:.4f}" for r in rows[-3:]), flush=True)

df = pd.DataFrame(rows)
csv = os.path.join(a.out, "results.csv")
df.to_csv(csv, index=False)

print("\n=== accuracy por brazo ===")
print(df.groupby(["dataset", "arm", "classifier"]).acc.agg(["mean", "std"]).round(4))

print("\n=== delta (random - zero), pareado por semilla ===")
piv = df.pivot_table(index=["dataset", "classifier", "run"], columns="arm", values="acc")
piv["delta"] = piv["random"] - piv["zero"]
print(piv.groupby(["dataset", "classifier"]).delta.agg(["mean", "std", "min", "max"]).round(4))
print("\nCSV ->", csv)
