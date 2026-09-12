"""Arm A (contaminated selection, published) vs Arm B (clean selection).
Same seeds, same everything else. Records best_it and downstream accuracy."""
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
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/critic_clean"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLFS = ["mlp", "rf", "xgboost"]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

rows = []
for run in range(a.runs):
    seed = 132 + run
    for arm, clean in [("contaminated", False), ("clean", True)]:
        set_seed(seed)                       # same seed for both arms
        data = prepare_db({"random_seed": seed, "device": DEV}, name=a.dataset)
        sdir = os.path.join(a.out, f"{arm}_run{run:02d}")
        cfg = {
            "device": DEV, "synth_type": "tame_critic", "ipc": a.ipc,
            "dm_iters": a.iters, "dm_lr": 0.5, "dm_batch_real": 128,
            "dm_embedder_type": "ln_res_l", "dm_embedder_size": "base",
            "dm_embed_hidden": 256, "dm_embed_dim": max(4, min(48, a.ipc - 2)),
            "dm_use_critic": True, "dm_adv_weight": 0.05, "dm_n_critic": 3,
            "dm_critic_clean_selection": clean,
            # Instrumentation only: recorded for offline validation-based
            # selection later, never used to pick best_syn here.
            "snapshot_every": 100, "return_snapshots": True,
            "save_dir": sdir, "random_seed": seed,
        }
        t0 = time.time()
        out = synthesize("tame_critic", data, cfg)
        dt = time.time() - t0
        X_syn, y_syn, snaps = out if len(out) == 3 else (*out, [])
        if snaps:
            torch.save({"snapshots": [(i, t.cpu()) for i, t in snaps],
                        "y_syn": y_syn.cpu()},
                       os.path.join(sdir, "snapshots.pt"))

        # best_it lands in the checkpoint the synthesizer writes
        ck = torch.load(os.path.join(sdir, "best_syn.pt"), map_location="cpu")
        best_it, best_loss = int(ck["best_it"]), float(ck["best_loss"])

        for clf in CLFS:
            set_seed(seed)
            td = {"X_train": X_syn, "y_train": y_syn,
                  "X_val": data["X_val"], "y_val": data["y_val"],
                  "input_dim": data["input_dim"], "num_classes": data["num_classes"]}
            m = train_classifier(td, {**cfg, "classifier": clf,
                                      "classifier_hidden": [128, 64], "classifier_epochs": 20})
            acc, auc = evaluate_classifier(m, data, DEV)
            rows.append(dict(dataset=a.dataset, arm=arm, run=run, seed=seed,
                             classifier=clf, acc=float(acc), auc=float(auc),
                             best_it=best_it, best_loss=best_loss,
                             n_snapshots=len(snaps), synth_s=round(dt, 1)))
        print(f"[{arm:12s} run{run:02d}] best_it={best_it:4d}  {dt:.1f}s  "
              + "  ".join(f"{r['classifier']}={r['acc']:.4f}" for r in rows[-3:]), flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "results.csv"), index=False)
print("\n=== best_it por brazo ===")
print(df.groupby("arm").best_it.describe()[["mean", "50%", "min", "max"]])
print("\n=== accuracy ===")
print(df.groupby(["arm", "classifier"]).acc.agg(["mean", "std"]).round(4))
print("\nCSV ->", os.path.join(a.out, "results.csv"))
