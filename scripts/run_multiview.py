#!/usr/bin/env python3
"""
Multi-view + checkpoint-selection experiment.

Question 1 (views): does matching moments under m simultaneous random
embedders per iteration improve the MEAN quality of a run and shrink the
across-seed variance? If yes, gains are real optimization gains; if the mean
stays flat and only best-of-K selection helps, the variance is mostly luck.

Question 2 (capacity): at high IPC the mean+cov objective is underdetermined
(O(p + p^2) constraints per class regardless of IPC). Two candidate fixes,
tested head to head at IPC=250: scale p (paper protocol, p=196) vs scale m
(more views at small p) vs both.

Question 3 (checkpointing): the current best-loss checkpoint compares losses
computed under DIFFERENT random embedders and minibatches, so it partly
selects lucky views. Using the snapshot trail of every run, compare four
within-run selection strategies:
    init        the starting real subset (free "random" reference)
    final       last iterate
    best_loss   what tame_synthesize returns today
    best_probe  snapshot with lowest fixed-probe moment score
    oracle      snapshot with highest test acc (upper bound, not a claim)

Outputs:
    results_multiview/runs.csv       one row per run x classifier (returned set)
    results_multiview/snapshots.csv  one row per snapshot x classifier
    results_multiview/summary printout
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
from synth.tame_synth import tame_synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier
from run_best_of_k import ProbeScorer, set_seed, eval_on_split


def main():
    DATASETS = ["adult", "electricity", "magic", "pendigits", "satimage"]
    # (ipc, embed_dim p, views m)
    CONDITIONS = [
        # IPC=10: does m help where TAME already wins?
        (10, 8, 1), (10, 8, 4), (10, 8, 8),
        # IPC=50: the crossover zone
        (50, 48, 1), (50, 48, 4), (50, 48, 8),
        # IPC=250: underdetermined regime — p-fix vs m-fix vs both
        (250, 48, 1), (250, 196, 1), (250, 48, 8), (250, 196, 8),
    ]
    K = 3
    CLASSIFIERS = ["rf", "xgboost"]
    BASE_SEED = 1000
    SNAPSHOT_EVERY = 100
    RESULTS_DIR = "results_multiview"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_rows, snap_rows = [], []

    def eval_set(X_syn, y_syn, data, config):
        """Train each classifier on the set; return {clf: (val, test)}."""
        train_data = {
            "X_train": X_syn, "y_train": y_syn,
            "X_val": data["X_val"], "y_val": data["y_val"],
            "input_dim": data["input_dim"],
            "num_classes": data["num_classes"],
        }
        out = {}
        for clf in CLASSIFIERS:
            cfg = dict(config)
            cfg["classifier"] = clf
            try:
                model = train_classifier(train_data, cfg)
                out[clf] = (eval_on_split(model, data, device, "val"),
                            eval_on_split(model, data, device, "test"))
            except Exception as e:
                print(f"    {clf} eval FAILED: {e}")
        return out

    for db in DATASETS:
        if db not in DATASET_REGISTRY:
            print(f"SKIP {db}")
            continue

        scorers = {}  # p -> ProbeScorer (probe dim must match embed_dim)

        for ipc, p, m in CONDITIONS:
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
                "dm_embed_dim": p,
                "dm_views": m,
                "snapshot_every": SNAPSHOT_EVERY,
                "return_snapshots": True,
                "classifier_hidden": [128, 64],
                "classifier_epochs": 20,
                "random_seed": BASE_SEED,
            }

            data0 = prepare_db(config, name=db)
            if p not in scorers:
                print(f"[{db}] building probe scorer (p={p}) ...")
                scorers[p] = ProbeScorer(data0, config, device)
            scorer = scorers[p]

            print(f"\n[{db}] ipc={ipc} p={p} m={m} K={K}")
            for k in range(K):
                run_seed = BASE_SEED + k
                set_seed(run_seed)
                cfg = dict(config)
                cfg["random_seed"] = run_seed

                data = prepare_db(cfg, name=db)

                t0 = time.time()
                try:
                    best_syn, y_syn, snapshots = tame_synthesize(data, cfg)
                except Exception as e:
                    print(f"  run{k} FAILED: {e}")
                    continue
                synth_time = time.time() - t0

                # ---- returned (best-loss) set: the headline number ----
                res = eval_set(best_syn, y_syn, data, cfg)
                probe_best = scorer.score(best_syn, y_syn)
                for clf, (v, t) in res.items():
                    run_rows.append({
                        "dataset": db, "ipc": ipc, "p": p, "m": m,
                        "run": k, "classifier": clf,
                        "val_acc": v, "test_acc": t,
                        "probe_score": probe_best,
                        "synth_time": synth_time,
                    })
                    print(f"  run{k} {clf}: test={t:.4f} "
                          f"probe={probe_best:.4f} time={synth_time:.0f}s")

                # ---- snapshot trail: checkpoint-selection study ----
                for it, snap in snapshots:
                    probe = scorer.score(snap, y_syn)
                    sres = eval_set(snap, y_syn, data, cfg)
                    for clf, (v, t) in sres.items():
                        snap_rows.append({
                            "dataset": db, "ipc": ipc, "p": p, "m": m,
                            "run": k, "iter": it, "classifier": clf,
                            "val_acc": v, "test_acc": t, "probe_score": probe,
                        })

                pd.DataFrame(run_rows).to_csv(
                    os.path.join(RESULTS_DIR, "runs_partial.csv"), index=False)
                pd.DataFrame(snap_rows).to_csv(
                    os.path.join(RESULTS_DIR, "snapshots_partial.csv"), index=False)

    runs = pd.DataFrame(run_rows)
    snaps = pd.DataFrame(snap_rows)
    runs.to_csv(os.path.join(RESULTS_DIR, "runs.csv"), index=False)
    snaps.to_csv(os.path.join(RESULTS_DIR, "snapshots.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/runs.csv ({len(runs)}), "
          f"snapshots.csv ({len(snaps)})")

    analyze(runs, snaps)


def analyze(runs, snaps):
    pd.set_option("display.width", 250)

    print("\n" + "=" * 100)
    print("  Q1/Q2 — mean and std of test acc by (ipc, p, m), averaged over datasets")
    print("=" * 100)
    agg = runs.groupby(["ipc", "p", "m", "classifier"])["test_acc"].agg(
        ["mean", "std"]).round(4)
    print(agg.to_string())

    print("\n" + "=" * 100)
    print("  Q3 — checkpoint selection strategies (test acc, averaged over datasets/runs)")
    print("=" * 100)
    strat_rows = []
    for (db, ipc, p, m, run, clf), g in snaps.groupby(
            ["dataset", "ipc", "p", "m", "run", "classifier"]):
        opt = g[g["iter"] >= 0]  # exclude the init snapshot from selection
        if opt.empty:
            continue
        init = g[g["iter"] == -1]
        strat_rows.append({
            "ipc": ipc, "p": p, "m": m, "classifier": clf,
            "init": init["test_acc"].iloc[0] if len(init) else np.nan,
            "final": opt.loc[opt["iter"].idxmax(), "test_acc"],
            "best_probe": opt.loc[opt["probe_score"].idxmin(), "test_acc"],
            "best_val": opt.loc[opt["val_acc"].idxmax(), "test_acc"],
            "oracle": opt["test_acc"].max(),
        })
    strat = pd.DataFrame(strat_rows)
    agg2 = strat.groupby(["ipc", "m", "classifier"])[
        ["init", "final", "best_probe", "best_val", "oracle"]].mean().round(4)
    print(agg2.to_string())
    print("\nNote: 'best_loss' strategy = the returned set in runs.csv; compare "
          "runs.test_acc against these columns at matching (ipc, p, m).")


if __name__ == "__main__":
    main()
