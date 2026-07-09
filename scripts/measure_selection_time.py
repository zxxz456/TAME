#!/usr/bin/env python3
"""
Measure the wall-clock cost of validation-based snapshot selection, for the
same (dataset, embedder) rows as the synthesis-time table (Table
tab:synth_time), to add a "Selection time" column.

Selection cost = for each candidate snapshot, train the downstream classifier
on it (IPC x C samples) and evaluate on the validation split. The full
selection scores N_SNAPSHOTS candidates per classifier. Note the cost depends
only on the distilled-set size and the validation set, NOT on the embedder or
on snapshot content, so the three `letter` rows should match (a sanity check).

To obtain a genuine distilled snapshot we run a short real TAME distillation
(SNAP_ITERS iters) with the specified embedder and time selection on its
output; timing is independent of iteration count since the set size is fixed.

Reported per row:
  sel_per_snap_{clf} : mean seconds to train {clf} on one snapshot + eval on val
  sel_time_{clf}     : N_SNAPSHOTS * sel_per_snap_{clf}   (select for one clf)
  sel_time_total     : sum over the 3 classifiers          (select for the pool)

Output: results_selection_timing/results.csv
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prepare_database import prepare_db
from synth.tame_synth import tame_synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier

IPC = 50
N_SNAPSHOTS = 11          # snapshots scored in selection (iters 0..1000 / 100)
SNAP_ITERS = 100          # short distillation just to obtain a genuine snapshot
REPS = 5                  # timed repetitions per classifier (after 1 warmup)
CLASSIFIERS = ["mlp", "rf", "xgboost"]

# (dataset, embedder) matching Table tab:synth_time
ROWS = [
    ("adult", "ln_res_l"), ("electricity", "ln_res_l"), ("madelon", "ln_res_l"),
    ("magic", "ln_res_l"), ("phishing", "ln_res_l"), ("satimage", "ln_res_l"),
    ("letter", "dcnv2_base"), ("letter", "ln_res_l"), ("letter", "node"),
]

RESULTS_DIR = "results_selection_timing"
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def eval_val(model, data, device):
    vv = dict(data); vv["X_test"] = data["X_val"]; vv["y_test"] = data["y_val"]
    return evaluate_classifier(model, vv, device)


def time_selection(X_syn, y_syn, data, device, base_cfg):
    """Return {clf: mean seconds for one (train-on-snapshot + eval-on-val)}."""
    td = {"X_train": X_syn, "y_train": y_syn,
          "X_val": data["X_val"], "y_val": data["y_val"],
          "input_dim": data["input_dim"], "num_classes": data["num_classes"]}
    out = {}
    for clf in CLASSIFIERS:
        cfg = dict(base_cfg); cfg["classifier"] = clf
        # warmup (discarded): absorbs CUDA/library first-call overhead
        m = train_classifier(td, cfg); eval_val(m, data, device)
        times = []
        for _ in range(REPS):
            t0 = time.time()
            m = train_classifier(td, cfg)
            eval_val(m, data, device)
            times.append(time.time() - t0)
        out[clf] = float(np.mean(times))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # global warmup so the very first measured row isn't penalised
    set_seed(0)
    _w = prepare_db({"dataset_name": "magic", "device": device, "ipc": IPC,
                     "classifier_hidden": [128, 64], "classifier_epochs": 20,
                     "random_seed": 0}, name="magic")

    rows = []
    for ds, emb in ROWS:
        print(f"\n[{ds} x {emb}] distilling short snapshot (IPC={IPC}, {SNAP_ITERS} iters)...")
        set_seed(1)
        cfg = {
            "dataset_name": ds, "device": device, "synth_type": "tame", "ipc": IPC,
            "dm_iters": SNAP_ITERS, "dm_lr": 0.5, "dm_batch_real": 128,
            "dm_embedder_type": emb, "dm_embedder_size": "base",
            "dm_embed_hidden": 256, "dm_embed_dim": max(4, min(48, IPC - 2)),
            "classifier_hidden": [128, 64], "classifier_epochs": 20, "random_seed": 1,
        }
        data = prepare_db(cfg, name=ds)
        X_syn, y_syn = tame_synthesize(data, cfg)  # genuine distilled set

        sel = time_selection(X_syn, y_syn, data, device, cfg)
        C = data["num_classes"]
        per_snap = {c: sel[c] for c in CLASSIFIERS}
        sel_time = {c: N_SNAPSHOTS * sel[c] for c in CLASSIFIERS}
        total = sum(sel_time.values())

        row = {
            "dataset": ds, "embedder": emb, "classes": C,
            "features": data["input_dim"], "train_samples": len(data["y_train"]),
            "n_snapshots": N_SNAPSHOTS,
            **{f"sel_per_snap_{c}": round(per_snap[c], 4) for c in CLASSIFIERS},
            **{f"sel_time_{c}": round(sel_time[c], 3) for c in CLASSIFIERS},
            "sel_time_total": round(total, 3),
        }
        rows.append(row)
        print(f"  per-snapshot: {[(c, round(per_snap[c],3)) for c in CLASSIFIERS]}")
        print(f"  sel_time (11 snap): "
              f"{[(c, round(sel_time[c],2)) for c in CLASSIFIERS]}  total={total:.2f}s")

        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/results.csv")
    print("\n=== summary (Selection time = sel_time_total, seconds) ===")
    print(df[["dataset", "embedder", "classes", "features", "train_samples",
              "sel_time_mlp", "sel_time_rf", "sel_time_xgboost",
              "sel_time_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
