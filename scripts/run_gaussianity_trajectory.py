#!/usr/bin/env python3
"""
Gaussianity of the distilled set as a function of distillation iteration.

Re-runs the same distillations as run_val_checkpoint_eval.py (same seeds ->
same trajectories) but instead of training classifiers, measures normality
of every snapshot in DATA space using the paper's own metrics
(Shapiro-Wilk pass rate per class; Mardia kurtosis deviation where the
per-class sample size allows it).

This is the missing mechanistic link for the checkpointing story:
if pass rate rises with iteration while tree accuracy peaks early on
categorical/imbalanced datasets, the Gaussianization account is confirmed
by direct measurement rather than inference.

Output: results_gaussianity_traj/results.csv
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.prepare_database import prepare_db, DATASET_REGISTRY
from synth.tame_synth import tame_synthesize
from run_normality_test import analyze_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    DATASETS = ["adult", "airlines", "bank", "climate", "credit", "electricity",
                "german", "higgs", "letter", "madelon", "magic", "pageblocks",
                "pendigits", "phishing", "satimage", "segment", "shuttle", "spambase"]
    IPCS = [10, 50]
    K = 3
    BASE_SEED = 1000  # must match run_val_checkpoint_eval.py
    RESULTS_DIR = "results_gaussianity_traj"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for db in DATASETS:
        if db not in DATASET_REGISTRY:
            print(f"SKIP {db}")
            continue
        for ipc in IPCS:
            print(f"\n[{db}] ipc={ipc} K={K}")
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
                    "snapshot_every": 100,
                    "return_snapshots": True,
                    "random_seed": run_seed,
                }
                data = prepare_db(config, name=db)

                t0 = time.time()
                try:
                    _, y_syn, snapshots = tame_synthesize(data, config)
                except Exception as e:
                    print(f"  run{k} FAILED: {e}")
                    continue

                y_np = y_syn.cpu().numpy()
                for it, snap in snapshots:
                    X_np = snap.cpu().numpy()
                    try:
                        m = analyze_matrix(X_np, y_np)
                    except Exception as e:
                        print(f"  run{k} iter{it} normality FAILED: {e}")
                        continue
                    rows.append({
                        "dataset": db, "ipc": ipc, "run": k, "iter": it,
                        "shapiro_pass_rate": m.get("shapiro_pass_rate"),
                        "mardia_kurt_deviation": m.get("mardia_kurt_deviation"),
                        "mardia_kurt_stat": m.get("mardia_kurt_stat"),
                    })
                print(f"  run{k}: {len(snapshots)} snapshots analyzed "
                      f"({time.time()-t0:.0f}s)")

                pd.DataFrame(rows).to_csv(
                    os.path.join(RESULTS_DIR, "partial.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/results.csv ({len(df)} rows)")

    pd.set_option("display.width", 200)
    print("\n=== mean Shapiro-Wilk pass rate by iteration ===")
    piv = df.pivot_table(index="ipc", columns="iter",
                         values="shapiro_pass_rate", aggfunc="mean").round(3)
    print(piv.to_string())


if __name__ == "__main__":
    main()
