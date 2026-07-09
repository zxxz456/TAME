#!/usr/bin/env python3
"""
Sensitivity sweep for the covariance weight (lambda / cov_weight in Eq. 7):

    L = ||mu_T - mu_S||^2 + lambda * ||Sigma_T - Sigma_S||^2_F

All published results use lambda = 1.0, untuned. Reviewer 3 asks how
sensitive TAME is to this choice and whether adaptive schemes were
considered. This sweep answers empirically: accuracy vs lambda over two
orders of magnitude, at two IPC/embedding-dim regimes (the effective
mean/cov balance shifts with p, so the optimum may move between them).

Protocol matches the paper: best-loss return, no snapshot selection.
Output: results_lambda_sweep/results.csv
"""

import os
import sys
import random
import time
import numpy as np
import pandas as pd
import torch

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prepare_database import prepare_db, DATASET_REGISTRY
from synth.registry import synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    DATASETS = ["adult", "electricity", "magic", "pendigits", "satimage"]
    IPCS = [10, 50]
    LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0]
    K = 3
    CLASSIFIERS = ["mlp", "rf", "xgboost"]
    BASE_SEED = 1000
    RESULTS_DIR = "results_lambda_sweep"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []

    for db in DATASETS:
        if db not in DATASET_REGISTRY:
            print(f"SKIP {db}")
            continue
        for ipc in IPCS:
            for lam in LAMBDAS:
                accs = {clf: [] for clf in CLASSIFIERS}
                times = []
                print(f"\n[{db}] ipc={ipc} lambda={lam} K={K}")
                for k in range(K):
                    run_seed = BASE_SEED + k
                    set_seed(run_seed)
                    config = {
                        "dataset_name": db,
                        "device": device,
                        "synth_type": "tame",
                        "ipc": ipc,
                        "dm_iters": 1000,
                        "dm_lr": 0.5,
                        "dm_batch_real": 128,
                        "dm_embedder_type": "ln_res_l",
                        "dm_embedder_size": "base",
                        "dm_embed_hidden": 256,
                        "dm_embed_dim": max(4, min(48, ipc - 2)),
                        "cov_weight": lam,
                        "classifier_hidden": [128, 64],
                        "classifier_epochs": 20,
                        "random_seed": run_seed,
                    }
                    data = prepare_db(config, name=db)
                    t0 = time.time()
                    try:
                        X_syn, y_syn = synthesize("tame", data, config)
                    except Exception as e:
                        print(f"  run{k} FAILED: {e}")
                        continue
                    times.append(time.time() - t0)

                    train_data = {
                        "X_train": X_syn, "y_train": y_syn,
                        "X_val": data["X_val"], "y_val": data["y_val"],
                        "input_dim": data["input_dim"],
                        "num_classes": data["num_classes"],
                    }
                    for clf in CLASSIFIERS:
                        cfg = dict(config)
                        cfg["classifier"] = clf
                        try:
                            model = train_classifier(train_data, cfg)
                            acc, _ = evaluate_classifier(model, data, device)
                            accs[clf].append(float(acc))
                        except Exception as e:
                            print(f"  run{k} {clf} FAILED: {e}")

                for clf in CLASSIFIERS:
                    if not accs[clf]:
                        continue
                    row = {
                        "dataset": db, "ipc": ipc, "cov_weight": lam,
                        "classifier": clf, "num_runs": len(accs[clf]),
                        "test_acc_mean": float(np.mean(accs[clf])),
                        "test_acc_std": float(np.std(accs[clf])),
                        "synth_time_mean": (float(np.mean(times))
                                            if times else float("nan")),
                    }
                    all_rows.append(row)
                    print(f"  {clf}: {row['test_acc_mean']:.4f} "
                          f"+/- {row['test_acc_std']:.4f}")

                pd.DataFrame(all_rows).to_csv(
                    os.path.join(RESULTS_DIR, "partial.csv"), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/results.csv ({len(df)} rows)")

    pd.set_option("display.width", 250)
    print("\n=== mean test acc by lambda (averaged over datasets) ===")
    piv = df.pivot_table(index=["ipc", "classifier"], columns="cov_weight",
                         values="test_acc_mean", aggfunc="mean").round(4)
    print(piv.to_string())


if __name__ == "__main__":
    main()
