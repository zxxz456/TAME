#!/usr/bin/env python3
import os
import random
import time
import numpy as np
import pandas as pd
import torch

from data.prepare_database import prepare_db, DATASET_REGISTRY
from synth.registry import synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_synth_data(X_syn, y_syn, out_dir, dataset, method, embedder, ipc, run_id,
                    snapshots=None):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{dataset}__{method}__{embedder}__ipc{ipc}__run{run_id:02d}"
    payload = {"X_syn": X_syn.cpu(), "y_syn": y_syn.cpu()}
    if snapshots:
        # full trail (init at iter -1 through final): any selector or metric
        # can be recomputed offline from these
        payload["snapshots"] = [(it, s.cpu()) for it, s in snapshots]
    torch.save(payload, os.path.join(out_dir, f"{tag}.pt"))


def _eval_val_acc(model, data, device):
    """Validation accuracy via the same evaluation path as test."""
    val_view = dict(data)
    val_view["X_test"] = data["X_val"]
    val_view["y_test"] = data["y_val"]
    acc, _ = evaluate_classifier(model, val_view, device)
    return float(acc)


def run_experiment(config, num_runs=10):
    classifiers = config.get("classifiers", ["mlp"])
    device = config["device"]
    synth_dir = config.get("synth_save_dir", "synth_outputs")
    # validation-selected checkpointing: candidates = init + snapshots
    # (+ the best-loss iterate), selected per classifier by validation
    # accuracy. Only synthesizers that support snapshots participate.
    val_checkpoint = bool(config.get("val_checkpoint", False))

    accs = {clf: [] for clf in classifiers}          # val-selected
    accs_bl = {clf: [] for clf in classifiers}       # best-loss (paper method)
    synth_times = []

    for run_id in range(num_runs):
        run_seed = config.get("random_seed", 132) + run_id
        set_seed(run_seed)

        data = prepare_db(config, name=config["dataset_name"])

        # advance the per-run seed inside the config too: reference
        # synthesizers (leverage/vq/random) re-seed internally from
        # config["random_seed"], so without this all runs would be identical
        run_config = dict(config)
        run_config["random_seed"] = run_seed

        t0 = time.time()
        out = synthesize(
            synth_type=run_config["synth_type"], data=data, config=run_config
        )
        synth_times.append(time.time() - t0)

        snapshots = []
        if len(out) == 3:
            X_syn, y_syn, snapshots = out
        else:
            X_syn, y_syn = out

        save_synth_data(
            X_syn, y_syn, synth_dir,
            config["dataset_name"], config["synth_type"],
            config.get("dm_embedder_type", "none"), config["ipc"], run_id,
            snapshots=snapshots,
        )

        # candidate sets: best-loss output first, then init + snapshots
        candidates = [X_syn] + [s for _, s in snapshots]

        for clf in classifiers:
            clf_config = dict(run_config)
            clf_config["classifier"] = clf

            best_val, best_model, bl_test = -1.0, None, None
            for idx, X_cand in enumerate(candidates):
                train_data = {
                    "X_train": X_cand,
                    "y_train": y_syn,
                    "X_val": data["X_val"],
                    "y_val": data["y_val"],
                    "input_dim": data["input_dim"],
                    "num_classes": data["num_classes"],
                }
                model = train_classifier(train_data, clf_config)
                if idx == 0:
                    # the best-loss iterate = the paper's published method
                    a, _ = evaluate_classifier(model, data, device)
                    bl_test = float(a)
                val = (_eval_val_acc(model, data, device)
                       if len(candidates) > 1 else 0.0)
                if (model is not None and val > best_val) or best_model is None:
                    best_val, best_model = val, model

            # test is read once, after the selection is frozen
            best_test, _ = evaluate_classifier(best_model, data, device)
            best_test = float(best_test)

            accs[clf].append(best_test)
            accs_bl[clf].append(bl_test)
            print(
                f"[{config['dataset_name']} | {config['synth_type']} | "
                f"{config.get('dm_embedder_type', '')} | run {run_id:02d} | "
                f"{clf}] acc={best_test:.4f}"
                + (f" (val-selected of {len(candidates)})"
                   if len(candidates) > 1 else "")
            )

    rows = []
    for clf in classifiers:
        row = {
            "dataset": config["dataset_name"],
            "method": config["synth_type"],
            "embedder": config.get("dm_embedder_type", ""),
            "ipc": config["ipc"],
            "classifier": clf,
            "num_runs": num_runs,
            "test_acc_mean": float(np.mean(accs[clf])),
            "test_acc_std": float(np.std(accs[clf])),
            "test_acc_bestloss_mean": float(np.mean(accs_bl[clf])),
            "test_acc_bestloss_std": float(np.std(accs_bl[clf])),
            "synth_time_mean": float(np.mean(synth_times)),
        }
        rows.append(row)
        print(
            f"[SUMMARY | {row['dataset']} | {row['method']} | {clf}] "
            f"acc={row['test_acc_mean']:.4f}±{row['test_acc_std']:.4f} | "
            f"time={row['synth_time_mean']:.2f}s"
        )
    return rows


def main():
    # higgs_1m validation sweep: all three embedders, 5 runs, both IPCs,
    # val-checkpointing on, snapshot trails + dual accuracies recorded.
    # full airline re-run with current (dual-accuracy, seed-fixed) code:
    # ln_res_l TAME gives best-loss AND best-val; baselines get error bars.
    DB_LIST = ["airline_satisfaction"]
    SYNTH_TYPES = ["tame", "leverage_score", "vq", "random", "full"]
    IPCs = [10, 50]
    EMBEDDERS = ["ln_res_l"]
    CLASSIFIERS = ["mlp", "rf", "xgboost"]
    NUM_RUNS = 10

    RESULTS_DIR = "results_final_airline2"
    SYNTH_DIR = "synth_final_airline2"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SYNTH_DIR, exist_ok=True)

    all_rows = []

    no_embedder = {
        "ctgan", "tvae", "full", "random", "vq", "voronoi", "gonzalez", "leverage_score"}

    for db in DB_LIST:
        for synth_type in SYNTH_TYPES:
            embedder_iter = EMBEDDERS if synth_type not in no_embedder else [""]

            for embedder in embedder_iter:
                for ipc in IPCs:
                    config = {
                        "dataset_name": db,
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "synth_type": synth_type,
                        "ipc": ipc,

                        "dm_iters": 1000,
                        "dm_lr": 0.5,
                        "dm_batch_real": 128,
                        "dm_embedder_type": embedder,
                        "dm_embedder_size": "base",
                        "dm_embed_hidden": 256,
                        # rank-safe embedding dim (Sec 4.5): p < IPC
                        "dm_embed_dim": max(4, min(48, ipc - 2)),

                        # validation-selected checkpointing (only tame
                        # returns snapshots; baselines are unaffected)
                        "val_checkpoint": True,
                        "snapshot_every": 100,
                        "return_snapshots": True,

                        "ctgan_epochs": 100,
                        "tvae_epochs": 100,

                        "classifiers": CLASSIFIERS,
                        "classifier_hidden": [128, 64],
                        "classifier_epochs": 20,
                        "random_seed": 132,
                        "synth_save_dir": SYNTH_DIR,
                    }

                    # full-data baseline is ~deterministic and expensive on
                    # the large datasets: 3 runs suffice
                    n_runs = 3 if synth_type == "full" else NUM_RUNS
                    rows = run_experiment(config, num_runs=n_runs)
                    all_rows.extend(rows)
                    # incremental save: a crash never loses finished work
                    pd.DataFrame(all_rows).to_csv(
                        os.path.join(RESULTS_DIR, "partial.csv"), index=False)

    master = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "results.csv")
    master.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(master)} rows)")


if __name__ == "__main__":
    main()
