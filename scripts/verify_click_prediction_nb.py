#!/usr/bin/env python3
"""
Standalone, from-scratch verification of the click_prediction_small cell that
looked most suspicious in results_tdbench_valcheckpoint_p48. Independent of
the main TDBench sweep script: fresh seeds, its own data load, its own
snapshot/selection logic re-derived from first principles (not imported from
run_tdbench_valcheckpoint.py), so a bug shared by both wouldn't hide here.

Reports, for every run: acc_full, acc_random@IPC=10, best-loss distilled
accuracy, best-val distilled accuracy -- for ALL 5 classifiers (not just NB),
so the whole row can be cross-checked, not just the one flagged cell.

Output: results_verify_click_pred/results.csv
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prepare_database import prepare_db, DATASET_REGISTRY
from data.tdbench_datasets import register_tdbench_datasets
from synth.tame_synth import tame_synthesize
from models.classifiers import train_classifier

DATASET = "click_prediction_small"
IPC = 10
EMBED_DIM = 48
CLFS = ["mlp_sci", "lr", "knn", "nb", "xgboost"]
N_RUNS = 8              # more than the original 5, independent seeds
BASE_SEED = 9000        # disjoint from the original run's seed range (42-46)
RESULTS_DIR = "results_verify_click_pred"
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def predict(model, X, device):
    if hasattr(model, "predict"):
        Xn = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        return model.predict(Xn)
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device).float())
        if logits.dim() == 1 or logits.shape[-1] == 1:
            preds = (torch.sigmoid(logits.view(-1)) >= 0.5).long()
        else:
            preds = logits.argmax(dim=-1)
    return preds.cpu().numpy()


def bacc(model, X, y, device):
    return float(balanced_accuracy_score(y.cpu().numpy(), predict(model, X, device)))


def train_eval_all(X, y, data, cfg, device):
    td = {"X_train": X, "y_train": y, "X_val": data["X_val"], "y_val": data["y_val"],
          "input_dim": data["input_dim"], "num_classes": data["num_classes"]}
    out = {}
    for clf in CLFS:
        c2 = dict(cfg); c2["classifier"] = clf
        m = train_classifier(td, c2)
        out[clf] = (bacc(m, data["X_val"], data["y_val"], device),
                    bacc(m, data["X_test"], data["y_test"], device))
    return out


def main():
    register_tdbench_datasets()
    assert DATASET in DATASET_REGISTRY, f"{DATASET} not registered"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for run in range(N_RUNS):
        seed = BASE_SEED + run
        set_seed(seed)
        cfg = {
            "dataset_name": DATASET, "device": device, "ipc": IPC,
            "dm_iters": 1000, "dm_lr": 0.5, "dm_batch_real": 128,
            "dm_embedder_type": "ln_res_l", "dm_embedder_size": "base",
            "dm_embed_hidden": 256, "dm_embed_dim": EMBED_DIM,
            "snapshot_every": 100, "return_snapshots": True,
            "classifier_hidden": [128, 64], "classifier_epochs": 20,
            "random_seed": seed,
        }
        data = prepare_db(cfg, name=DATASET)

        # full-data and random@IPC=10 baselines, independently computed here
        full_res = {}
        for clf in CLFS:
            c2 = dict(cfg); c2["classifier"] = clf
            m = train_classifier({"X_train": data["X_train"], "y_train": data["y_train"],
                                   "X_val": data["X_val"], "y_val": data["y_val"],
                                   "input_dim": data["input_dim"],
                                   "num_classes": data["num_classes"]}, c2)
            full_res[clf] = bacc(m, data["X_test"], data["y_test"], device)

        rng = np.random.default_rng(seed)
        Xtr, ytr = data["X_train"].cpu().numpy(), data["y_train"].cpu().numpy()
        idx = []
        for c in np.unique(ytr):
            ci = np.where(ytr == c)[0]
            idx.extend(rng.choice(ci, min(IPC, len(ci)), replace=False))
        Xr = torch.tensor(Xtr[idx], device=device, dtype=torch.float32)
        yr = torch.tensor(ytr[idx], device=device, dtype=torch.long)
        rand_res = {}
        for clf in CLFS:
            c2 = dict(cfg); c2["classifier"] = clf
            m = train_classifier({"X_train": Xr, "y_train": yr,
                                   "X_val": data["X_val"], "y_val": data["y_val"],
                                   "input_dim": data["input_dim"],
                                   "num_classes": data["num_classes"]}, c2)
            rand_res[clf] = bacc(m, data["X_test"], data["y_test"], device)

        # TAME distillation with snapshots
        best_syn, y_syn, snapshots = tame_synthesize(data, cfg)

        # best-loss (returned) accuracy
        bl_res = train_eval_all(best_syn, y_syn, data, cfg, device)

        # best-val: score every snapshot per classifier, pick by val bacc
        cand_scores = {clf: [] for clf in CLFS}
        for it, snap in snapshots:
            res = train_eval_all(snap, y_syn, data, cfg, device)
            for clf, (v, t) in res.items():
                cand_scores[clf].append((v, t, it))

        for clf in CLFS:
            v, t, it = max(cand_scores[clf], key=lambda x: x[0])
            rows.append({
                "run": run, "seed": seed, "classifier": clf,
                "acc_full": full_res[clf], "acc_random": rand_res[clf],
                "acc_bestloss": bl_res[clf][1], "acc_bestval": t,
                "bestval_iter": it,
            })
            print(f"run{run} {clf:8s}: full={full_res[clf]:.4f} "
                  f"random={rand_res[clf]:.4f} bestloss={bl_res[clf][1]:.4f} "
                  f"bestval={t:.4f}@{it}")

        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "partial.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/results.csv ({len(df)} rows)")

    print("\n=== summary (mean over {} independent runs) ===".format(N_RUNS))
    agg = df.groupby("classifier")[["acc_full", "acc_random", "acc_bestloss",
                                    "acc_bestval"]].agg(["mean", "std"]).round(4)
    print(agg.to_string())
    print("\n=== original run values (for comparison) ===")
    print("nb: full=0.516 random=0.541 bestloss=0.517 bestval=0.565")


if __name__ == "__main__":
    main()
