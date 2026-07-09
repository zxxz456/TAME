#!/usr/bin/env python3
"""
Validation-checkpointed TAME vs current best-loss return, now including MLP.

For each run: distill with snapshots every 100 iters, train each classifier
on every snapshot, select the snapshot per classifier by VALIDATION accuracy,
report its TEST accuracy. Columns: init / final / best_loss / best_val.

Output: results_val_checkpoint/results.csv + summary printout.
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
from synth.tame_synth import tame_synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_on_split(model, data, device, split):
    eval_data = dict(data)
    if split == "val":
        eval_data["X_test"] = data["X_val"]
        eval_data["y_test"] = data["y_val"]
    acc, _ = evaluate_classifier(model, eval_data, device)
    return float(acc)


def _predict(model, X, device):
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


def eval_bacc(model, data, device, split):
    from sklearn.metrics import balanced_accuracy_score
    X = data["X_val"] if split == "val" else data["X_test"]
    y = (data["y_val"] if split == "val" else data["y_test"]).cpu().numpy()
    return float(balanced_accuracy_score(y, _predict(model, X, device)))


def main():
    DATASETS = ["adult", "airlines", "bank", "climate", "credit", "electricity",
                "german", "higgs", "letter", "madelon", "magic", "pageblocks",
                "pendigits", "phishing", "satimage", "segment", "shuttle", "spambase",
                "airline_satisfaction"]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipcs", type=str, default="10,50",
                    help="comma-separated IPC values, e.g. --ipcs 10")
    ap.add_argument("--embed-dim", type=int, default=None,
                    help="fixed embedder output dim (default: ipc-scaled)")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the results dir, e.g. --tag p48")
    args = ap.parse_args()

    IPCS = [int(x) for x in args.ipcs.split(",")]
    EMBED_DIM = args.embed_dim
    K = 3
    CLASSIFIERS = ["mlp", "rf", "xgboost"]
    BASE_SEED = 1000
    RESULTS_DIR = "results_val_checkpoint" + (f"_{args.tag}" if args.tag else "")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []
    snap_rows = []  # per-snapshot trajectory log for init-to-convergence figures

    # resume: keep rows from a previous run and skip its completed datasets;
    # delete partial.csv to force a full rerun
    partial_path = os.path.join(RESULTS_DIR, "partial.csv")
    snap_partial_path = os.path.join(RESULTS_DIR, "snapshots_partial.csv")
    if os.path.exists(partial_path):
        prev = pd.read_csv(partial_path)
        # completeness is judged WITHIN the requested IPC set: a dataset is
        # done only if all its rows for these IPCs exist. Rows for other
        # IPCs are always preserved.
        expected = len(IPCS) * K * len(CLASSIFIERS)
        in_scope = prev[prev["ipc"].isin(IPCS)]
        counts = in_scope.groupby("dataset").size()
        done = {d for d, c in counts.items() if c >= expected}
        incomplete = (set(DATASETS) - done)
        redo = {d for d in incomplete if d in set(counts.index)}
        if redo:
            print(f"[resume] dropping in-scope rows of incomplete datasets "
                  f"for redo: {sorted(redo)}")
            prev = prev[~(prev["dataset"].isin(redo)
                          & prev["ipc"].isin(IPCS))]
        all_rows = prev.to_dict("records")
        if os.path.exists(snap_partial_path):
            sprev = pd.read_csv(snap_partial_path)
            sprev = sprev[~(sprev["dataset"].isin(redo)
                            & sprev["ipc"].isin(IPCS))]
            snap_rows = sprev.to_dict("records")
        DATASETS = [d for d in DATASETS if d not in done]
        print(f"[resume] {len(prev)} rows kept; running: {DATASETS}")

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
                    "dm_embed_dim": (EMBED_DIM if EMBED_DIM is not None
                                     else max(4, min(48, ipc - 2))),
                    "snapshot_every": 100,
                    "return_snapshots": True,
                    "classifier_hidden": [128, 64],
                    "classifier_epochs": 20,
                    "random_seed": run_seed,
                }
                data = prepare_db(config, name=db)

                t0 = time.time()
                try:
                    best_loss_syn, y_syn, snapshots = tame_synthesize(data, config)
                except Exception as e:
                    print(f"  run{k} FAILED: {e}")
                    continue
                synth_time = time.time() - t0

                # evaluate every candidate set per classifier
                candidates = [("best_loss", best_loss_syn)] + [
                    (f"iter_{it}", s) for it, s in snapshots]

                per_clf = {clf: {} for clf in CLASSIFIERS}
                for name, X_syn in candidates:
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
                            v = eval_on_split(model, data, device, "val")
                            t = eval_on_split(model, data, device, "test")
                            tb = eval_bacc(model, data, device, "test")
                            per_clf[clf][name] = (v, t, tb)
                            snap_rows.append({
                                "dataset": db, "ipc": ipc, "run": k,
                                "classifier": clf, "candidate": name,
                                "iter": (int(name.split("_")[1])
                                         if name.startswith("iter_") else -2),
                                "val_acc": v, "test_acc": t, "test_bacc": tb,
                            })
                        except Exception as e:
                            print(f"  run{k} {clf} {name} FAILED: {e}")

                for clf in CLASSIFIERS:
                    res = per_clf[clf]
                    opt_iters = {n: v for n, v in res.items()
                                 if n.startswith("iter_") and n != "iter_-1"}
                    if not opt_iters or "iter_-1" not in res:
                        continue
                    sel = max(opt_iters, key=lambda n: opt_iters[n][0])
                    nan3 = (np.nan, np.nan, np.nan)
                    row = {
                        "dataset": db, "ipc": ipc, "run": k, "classifier": clf,
                        "init": res["iter_-1"][1],
                        "final": res.get("iter_1000", nan3)[1],
                        "best_loss": res.get("best_loss", nan3)[1],
                        "best_val": opt_iters[sel][1],
                        "best_val_iter": int(sel.split("_")[1]),
                        "init_bacc": res["iter_-1"][2],
                        "best_loss_bacc": res.get("best_loss", nan3)[2],
                        "best_val_bacc": opt_iters[sel][2],
                        "synth_time": synth_time,
                    }
                    all_rows.append(row)
                    print(f"  run{k} {clf}: init={row['init']:.4f} "
                          f"best_loss={row['best_loss']:.4f} "
                          f"best_val={row['best_val']:.4f}@{row['best_val_iter']}")

                pd.DataFrame(all_rows).to_csv(
                    os.path.join(RESULTS_DIR, "partial.csv"), index=False)
                pd.DataFrame(snap_rows).to_csv(
                    os.path.join(RESULTS_DIR, "snapshots_partial.csv"), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    pd.DataFrame(snap_rows).to_csv(
        os.path.join(RESULTS_DIR, "snapshots.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/results.csv ({len(df)} rows), "
          f"snapshots.csv ({len(snap_rows)} rows)")

    pd.set_option("display.width", 200)
    print("\n=== mean / median across datasets ===")
    agg = df.groupby(["ipc", "classifier"])[
        ["init", "final", "best_loss", "best_val"]].agg(["mean", "median"]).round(4)
    print(agg.to_string())


if __name__ == "__main__":
    main()
