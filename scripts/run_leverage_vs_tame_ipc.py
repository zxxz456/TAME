#!/usr/bin/env python3
"""
Targeted check: does TAME's gap vs. Leverage Score on tree-based classifiers
(RF, XGBoost) change with IPC budget?

Motivation: Table 3/4 in the paper show Leverage Score matching or beating
TAME-LnRes on RF/XGBoost at IPC=50 for several datasets. This script sweeps
IPC to see whether TAME's relative position improves, worsens, or stays flat
as the synthetic budget grows.

Covariance loss requires embed_dim < ipc to avoid rank deficiency, so
embed_dim is scaled down for small IPC.
"""

import os
import sys
import random
import time
import numpy as np
import pandas as pd
import torch

# Windows: redirected stdout defaults to cp1252, which crashes on unicode
# chars printed anywhere in the pipeline. Force utf-8, never die on a print.
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


def embed_dim_for_ipc(ipc):
    # keep p comfortably below ipc to avoid covariance rank deficiency
    return max(4, min(48, ipc - 2))


def run_one(config, num_runs):
    device = config["device"]
    classifiers = config["classifiers"]
    accs = {clf: [] for clf in classifiers}

    for run_id in range(num_runs):
        run_seed = config.get("random_seed", 132) + run_id
        set_seed(run_seed)

        data = prepare_db(config, name=config["dataset_name"])

        t0 = time.time()
        X_syn, y_syn = synthesize(synth_type=config["synth_type"], data=data, config=config)
        synth_time = time.time() - t0

        train_data = {
            "X_train": X_syn, "y_train": y_syn,
            "X_val": data["X_val"], "y_val": data["y_val"],
            "input_dim": data["input_dim"], "num_classes": data["num_classes"],
        }

        for clf in classifiers:
            clf_config = dict(config)
            clf_config["classifier"] = clf
            model = train_classifier(train_data, clf_config)
            acc, _ = evaluate_classifier(model, data, device)
            accs[clf].append(float(acc))

    return accs, synth_time


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated dataset subset "
                         "(writes to results_leverage_vs_tame_ipc_extra/)")
    args = ap.parse_args()

    DATASETS = ["adult", "airlines", "bank", "climate", "credit", "electricity",
                "german", "higgs", "letter", "madelon", "magic", "pageblocks",
                "pendigits", "phishing", "satimage", "segment", "shuttle", "spambase"]
    RESULTS_DIR = "results_leverage_vs_tame_ipc"
    if args.only.strip():
        DATASETS = [d.strip() for d in args.only.split(",")]
        RESULTS_DIR += "_extra"
    IPCS = [10]
    CLASSIFIERS = ["rf", "xgboost"]
    NUM_RUNS = 3
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []

    for db in DATASETS:
        if db not in DATASET_REGISTRY:
            print(f"SKIP {db}")
            continue
        for ipc in IPCS:
            for synth_type, embedder in [("leverage_score", ""), ("tame", "ln_res_l")]:
                config = {
                    "dataset_name": db,
                    "device": device,
                    "synth_type": synth_type,
                    "ipc": ipc,
                    "dm_iters": 1000,
                    "dm_lr": 0.5,
                    "dm_batch_real": 128,
                    "dm_embedder_type": embedder,
                    "dm_embedder_size": "base",
                    "dm_embed_hidden": 256,
                    "dm_embed_dim": embed_dim_for_ipc(ipc),
                    "classifiers": CLASSIFIERS,
                    "classifier_hidden": [128, 64],
                    "classifier_epochs": 20,
                    "random_seed": 132,
                    "synth_save_dir": "synth_outputs_leverage_vs_tame_ipc",
                }
                print(f"\n[{db}] ipc={ipc} method={synth_type}{('/'+embedder) if embedder else ''}")
                try:
                    accs, synth_time = run_one(config, NUM_RUNS)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    continue

                for clf in CLASSIFIERS:
                    row = {
                        "dataset": db,
                        "method": synth_type,
                        "embedder": embedder,
                        "ipc": ipc,
                        "classifier": clf,
                        "num_runs": NUM_RUNS,
                        "test_acc_mean": float(np.mean(accs[clf])),
                        "test_acc_std": float(np.std(accs[clf])),
                        "synth_time": synth_time,
                    }
                    all_rows.append(row)
                    print(f"  {clf}: {row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}")

                pd.DataFrame(all_rows).to_csv(
                    os.path.join(RESULTS_DIR, "partial.csv"), index=False)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
