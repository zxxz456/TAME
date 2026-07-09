#!/usr/bin/env python3
"""
TDColER comparison with validation-checkpointed TAME (LnRes), IPC=10.

Same protocol as run_tdbench_comparison.py (23 TDBench datasets, balanced
accuracy, RR with the random@IPC=10 denominator, 5 runs), but each run's
distilled set is selected per classifier from the snapshot trail by
validation balanced accuracy, instead of returning the best-loss iterate.

Reference numbers (median RR at IPC=10, aggregated over classifiers):
    TAME-LnRes (paper, best-loss return)     0.465
    TDColER k-means alone                    0.665
    TDColER TF-SFT encoder                   0.615
    TDColER TF-SFT + k-means (best, 500 HPO) 0.406

Output: results_tdbench_valcheckpoint/results.csv
"""

import os
import sys
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prepare_database import prepare_db, DATASET_REGISTRY
from data.tdbench_datasets import register_tdbench_datasets, TDBENCH_NAMES
from synth.tame_synth import tame_synthesize
from models.classifiers import train_classifier

IPC = 10
NUM_RUNS = 5
CLASSIFIERS = ["mlp_sci", "rf", "knn", "lr", "nb", "xgboost"]
BASE_SEED = 42

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--only", type=str, default="",
                 help="comma-separated dataset subset (writes to _patch dir)")
_ap.add_argument("--embed-dim", type=int, default=8,
                 help="embedder output dim (8 = rank-safe at IPC=10; "
                      "the published Table 6 run used 48)")
_ap.add_argument("--tag", type=str, default="",
                 help="suffix for the results dir, e.g. --tag p48")
_args = _ap.parse_args()

ONLY = _args.only.strip()
EMBED_DIM = _args.embed_dim
RESULTS_DIR = ("results_tdbench_valcheckpoint" if not ONLY
               else "results_tdbench_valcheckpoint_patch")
if _args.tag:
    RESULTS_DIR = f"{RESULTS_DIR}_{_args.tag}"
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def bal_acc(model, X, y, device):
    return float(balanced_accuracy_score(
        y.cpu().numpy(), _predict(model, X, device)))


def train_eval_all(X_syn, y_syn, data, config, device):
    """Train each classifier on the set; return {clf: (val_bacc, test_bacc)}."""
    train_data = {
        "X_train": X_syn, "y_train": y_syn,
        "X_val": data["X_val"], "y_val": data["y_val"],
        "input_dim": data["input_dim"], "num_classes": data["num_classes"],
    }
    out = {}
    for clf in CLASSIFIERS:
        cfg = dict(config)
        cfg["classifier"] = clf
        try:
            model = train_classifier(train_data, cfg)
            out[clf] = (bal_acc(model, data["X_val"], data["y_val"], device),
                        bal_acc(model, data["X_test"], data["y_test"], device))
        except Exception as e:
            print(f"      {clf} failed: {e}")
    return out


def get_full_and_random(data, config, device):
    full = train_eval_all(data["X_train"], data["y_train"], data, config, device)

    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    classes = np.unique(y)
    rand = {clf: [] for clf in CLASSIFIERS}
    for run_id in range(NUM_RUNS):
        rng = np.random.default_rng(BASE_SEED + run_id)
        idxs = []
        for c in classes:
            ci = np.where(y == c)[0]
            idxs.extend(rng.choice(ci, min(IPC, len(ci)), replace=False))
        idxs = np.array(idxs)
        X_sub = torch.tensor(X[idxs], device=device, dtype=torch.float32)
        y_sub = torch.tensor(y[idxs], device=device, dtype=torch.long)
        res = train_eval_all(X_sub, y_sub, data, config, device)
        for clf, (_, t) in res.items():
            rand[clf].append(t)
    return ({c: v[1] for c, v in full.items()},
            {c: float(np.nanmean(v)) for c, v in rand.items() if v})


def main():
    register_tdbench_datasets()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_template = {
        "device": device,
        "ipc": IPC,
        "synth_type": "tame",
        "dm_iters": 1000,
        "dm_lr": 0.5,
        "dm_batch_real": 128,
        "dm_embedder_type": "ln_res_l",
        "dm_embedder_size": "base",
        "dm_embed_hidden": 256,
        "dm_embed_dim": EMBED_DIM,
        "snapshot_every": 100,
        "return_snapshots": True,
        "classifier_hidden": [128, 64],
        "classifier_epochs": 20,
        "random_seed": BASE_SEED,
    }

    # resume: skip datasets already completed in a previous (interrupted) run;
    # delete partial.csv to force a from-scratch run
    all_rows = []
    done = set()
    partial_path = os.path.join(RESULTS_DIR, "partial.csv")
    if os.path.exists(partial_path):
        prev = pd.read_csv(partial_path)
        all_rows = prev.to_dict("records")
        done = set(prev["dataset"].unique())
        print(f"[resume] found partial with {len(prev)} rows; "
              f"skipping completed datasets: {sorted(done)}")

    names = ([n for n in TDBENCH_NAMES if n in ONLY.split(",")]
             if ONLY else TDBENCH_NAMES)
    names = [n for n in names if n not in done]
    for db_name in names:
        if db_name not in DATASET_REGISTRY:
            print(f"  SKIP {db_name} (not registered)")
            continue
        print(f"\n{'='*70}\n  DATASET: {db_name}\n{'='*70}")

        config = dict(config_template)
        config["dataset_name"] = db_name
        try:
            data0 = prepare_db(config, name=db_name)
        except Exception as e:
            print(f"  FAILED to load {db_name}: {e}")
            continue

        print(f"  [Baselines] full + random@IPC={IPC}")
        full_accs, rand_accs = get_full_and_random(data0, config, device)
        for clf in CLASSIFIERS:
            if clf in full_accs:
                print(f"    {clf}: full={full_accs[clf]:.4f} "
                      f"random={rand_accs.get(clf, float('nan')):.4f}")

        # accumulate selected test acc per run per classifier, both selectors
        sel_val = {clf: [] for clf in CLASSIFIERS}   # val-checkpointed
        sel_loss = {clf: [] for clf in CLASSIFIERS}  # best-loss (paper baseline)
        synth_times = []

        for run_id in range(NUM_RUNS):
            run_seed = BASE_SEED + run_id
            set_seed(run_seed)
            cfg = dict(config)
            cfg["random_seed"] = run_seed

            data = prepare_db(cfg, name=db_name)
            t0 = time.time()
            try:
                best_loss_syn, y_syn, snapshots = tame_synthesize(data, cfg)
            except Exception as e:
                print(f"    run{run_id} distillation failed: {e}")
                continue
            synth_times.append(time.time() - t0)

            # best-loss baseline
            res_bl = train_eval_all(best_loss_syn, y_syn, data, cfg, device)
            for clf, (_, t) in res_bl.items():
                sel_loss[clf].append(t)

            # snapshot trail (exclude init from selection: iter >= 0)
            per_clf = {clf: [] for clf in CLASSIFIERS}
            for it, snap in snapshots:
                if it < 0:
                    continue
                res = train_eval_all(snap, y_syn, data, cfg, device)
                for clf, (v, t) in res.items():
                    per_clf[clf].append((v, t, it))
            for clf, cands in per_clf.items():
                if not cands:
                    continue
                v, t, it = max(cands, key=lambda x: x[0])
                sel_val[clf].append(t)
                print(f"    run{run_id} {clf}: best_val={t:.4f}@{it} "
                      f"(best_loss={res_bl.get(clf, (0, float('nan')))[1]:.4f})")

        for clf in CLASSIFIERS:
            if not sel_val[clf] or clf not in full_accs:
                continue
            full = full_accs[clf]
            rand10 = rand_accs.get(clf, float("nan"))
            denom = full - rand10
            for label, vals in [("val_checkpoint", sel_val[clf]),
                                ("best_loss", sel_loss[clf])]:
                if not vals:
                    continue
                mean_acc = float(np.nanmean(vals))
                rr = ((full - mean_acc) / denom
                      if np.isfinite(denom) and abs(denom) > 1e-6 else float("nan"))
                all_rows.append({
                    "dataset": db_name, "selector": label, "classifier": clf,
                    "ipc": IPC, "num_runs": len(vals),
                    "acc_full": full, "acc_random_ipc10": rand10,
                    "acc_distilled_mean": mean_acc,
                    "rr_tdcoler_formula": rr,
                    "synth_time_mean": (float(np.mean(synth_times))
                                        if synth_times else float("nan")),
                })

        pd.DataFrame(all_rows).to_csv(
            os.path.join(RESULTS_DIR, "partial.csv"), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/results.csv ({len(df)} rows)")

    print("\n" + "=" * 70)
    print("  MEDIAN RR BY SELECTOR (TDColER formula, IPC=10)")
    print("=" * 70)
    for label in ["best_loss", "val_checkpoint"]:
        sub = df[df["selector"] == label]
        rr_all = sub["rr_tdcoler_formula"].dropna()
        print(f"\n  {label}: overall median = {rr_all.median():.4f} (n={len(rr_all)})")
        for clf in CLASSIFIERS:
            rr = sub[sub["classifier"] == clf]["rr_tdcoler_formula"].dropna()
            if len(rr):
                print(f"    {clf:<8}: median={rr.median():.4f}")
    print("\n  References: TAME-LnRes paper 0.465 | TDColER best 0.406")


if __name__ == "__main__":
    main()
