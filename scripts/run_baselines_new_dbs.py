#!/usr/bin/env python3
"""
Reference baselines for the revision dataset(s) added to the benchmark
(currently: airline_satisfaction).

Methods: leverage_score, vq (k-means centroids), random, full.
IPC 10 and 50 (full runs once per dataset), classifiers MLP/RF/XGBoost,
3 seeds for the stochastic methods. Records plain AND balanced accuracy.

Output: results_baselines_new_dbs/results.csv
"""

import os
import sys
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prepare_database import prepare_db, DATASET_REGISTRY
from synth.registry import synthesize
from models.classifiers import train_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def main():
    DATASETS = ["airline_satisfaction"]
    METHODS = ["leverage_score", "vq", "random", "full"]
    IPCS = [10, 50]
    K = 3
    CLASSIFIERS = ["mlp", "rf", "xgboost"]
    BASE_SEED = 1000
    RESULTS_DIR = "results_baselines_new_dbs"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []

    # resume: skip datasets already completed in a previous run
    partial_path = os.path.join(RESULTS_DIR, "partial.csv")
    if os.path.exists(partial_path):
        prev = pd.read_csv(partial_path)
        done = set(prev["dataset"].unique())
        all_rows = prev.to_dict("records")
        DATASETS = [d for d in DATASETS if d not in done]
        print(f"[resume] {len(prev)} rows kept; running: {DATASETS}")

    for db in DATASETS:
        if db not in DATASET_REGISTRY:
            print(f"SKIP {db}")
            continue
        for ipc in IPCS:
            for method in METHODS:
                if method == "full" and ipc != IPCS[0]:
                    continue  # full data is IPC-independent, run once
                runs = 1 if method == "full" else K
                accs = {c: [] for c in CLASSIFIERS}
                baccs = {c: [] for c in CLASSIFIERS}
                print(f"\n[{db}] ipc={ipc} method={method} runs={runs}")
                for k in range(runs):
                    run_seed = BASE_SEED + k
                    set_seed(run_seed)
                    config = {
                        "dataset_name": db,
                        "device": device,
                        "synth_type": method,
                        "ipc": ipc,
                        # per-run seed: leverage/vq/random re-seed from this
                        "random_seed": run_seed,
                        "classifier_hidden": [128, 64],
                        "classifier_epochs": 20,
                    }
                    data = prepare_db(config, name=db)
                    try:
                        X_syn, y_syn = synthesize(method, data, config)
                    except Exception as e:
                        print(f"  run{k} synth FAILED: {e}")
                        continue
                    train_data = {
                        "X_train": X_syn, "y_train": y_syn,
                        "X_val": data["X_val"], "y_val": data["y_val"],
                        "input_dim": data["input_dim"],
                        "num_classes": data["num_classes"],
                    }
                    y_true = data["y_test"].cpu().numpy()
                    for clf in CLASSIFIERS:
                        cfg = dict(config)
                        cfg["classifier"] = clf
                        try:
                            model = train_classifier(train_data, cfg)
                            y_pred = predict(model, data["X_test"], device)
                            accs[clf].append(accuracy_score(y_true, y_pred))
                            baccs[clf].append(
                                balanced_accuracy_score(y_true, y_pred))
                        except Exception as e:
                            print(f"  run{k} {clf} FAILED: {e}")

                for clf in CLASSIFIERS:
                    if not accs[clf]:
                        continue
                    row = {
                        "dataset": db, "method": method, "ipc": ipc,
                        "classifier": clf, "num_runs": len(accs[clf]),
                        "test_acc_mean": float(np.mean(accs[clf])),
                        "test_acc_std": float(np.std(accs[clf])),
                        "test_bacc_mean": float(np.mean(baccs[clf])),
                        "test_bacc_std": float(np.std(baccs[clf])),
                    }
                    all_rows.append(row)
                    print(f"  {clf}: acc={row['test_acc_mean']:.4f} "
                          f"bacc={row['test_bacc_mean']:.4f}")

                pd.DataFrame(all_rows).to_csv(
                    os.path.join(RESULTS_DIR, "partial.csv"), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/results.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
