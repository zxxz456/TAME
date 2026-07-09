#!/usr/bin/env python3
"""
Best-of-K decoupling experiment: is TAME's run-to-run variance exploitable,
and if so, WHERE does it come from and HOW should the best run be selected?

Arms (all get the same K seeds):
  random           K random IPC subsets  -> "lucky subset" null hypothesis
  leverage_score   K leverage draws      -> the competitor, equal budget
  tame             init AND path vary    -> what you'd actually ship
  tame_fixed_init  same init subset for all K runs, only the optimization
                   path (embedder sequence, minibatches) varies
                   -> variance created by the optimization itself

Selectors, computed post-hoc from per-run records:
  best-by-val      pick run with highest validation accuracy (per classifier)
  best-by-probe    pick run with lowest probe-moment score (classifier-free)
  oracle           best test accuracy (upper bound; never a claim)

Diagnostics:
  spearman(val, test)          does validation rank-predict test across runs?
  spearman(-probe, test)       does the DM objective predict downstream acc?

Conclusions map:
  gain(tame) ~ gain(random)        -> selection is inherited init luck
  gain(tame_fixed_init) > 0        -> optimization variance is exploitable
  spearman(val,test) ~ 0           -> best-of-K claims are illusory, drop them
  spearman(-probe,test) > 0 on RF  -> moment objective predicts tree accuracy
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
from synth.tame_synth import cov_matrix
from models.embedders import sample_random_embedder
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def embed_dim_for_ipc(ipc):
    return max(4, min(48, ipc - 2))


def eval_on_split(model, data, device, split):
    eval_data = dict(data)
    if split == "val":
        eval_data["X_test"] = data["X_val"]
        eval_data["y_test"] = data["y_val"]
    acc, _ = evaluate_classifier(model, eval_data, device)
    return float(acc)


# ------------------------------------------------------------------
# Probe-moment scorer: deterministic, classifier-free set quality.
# Fixed-seed embedders + fixed real subsample -> same score for the
# same candidate set no matter when/where it is computed.
# ------------------------------------------------------------------

PROBE_SEEDS = [9000 + i for i in range(8)]
PROBE_SUBSAMPLE = 4096  # per class, fixed rng


class ProbeScorer:
    def __init__(self, data, config, device):
        self.device = device
        input_dim = int(data["input_dim"])
        num_classes = int(data["num_classes"])
        p = int(config["dm_embed_dim"])
        h = int(config["dm_embed_hidden"])

        X = data["X_train"].to(device).float()
        y = data["y_train"].cpu().numpy()
        sub_rng = np.random.default_rng(4242)

        self.num_classes = num_classes
        self.probes = []       # list of embedder nets
        self.real_moments = [] # per probe: list of (mu, cov) per class

        for seed in PROBE_SEEDS:
            set_seed(seed)
            net = sample_random_embedder(
                "ln_res_l", "base", input_dim, h, p, device)
            net.eval()
            moments = []
            with torch.no_grad():
                for c in range(num_classes):
                    idx = np.where(y == c)[0]
                    if len(idx) > PROBE_SUBSAMPLE:
                        idx = sub_rng.choice(idx, PROBE_SUBSAMPLE, replace=False)
                    z = net(X[idx])
                    moments.append(cov_matrix(z))
            self.probes.append(net)
            self.real_moments.append(moments)

    def score(self, X_syn, y_syn):
        """Lower = better moment match. Mean over probes and classes."""
        X_syn = X_syn.to(self.device).float()
        y_np = y_syn.cpu().numpy()
        total = 0.0
        with torch.no_grad():
            for net, moments in zip(self.probes, self.real_moments):
                for c in range(self.num_classes):
                    idx = np.where(y_np == c)[0]
                    if len(idx) == 0:
                        continue
                    mu_s, cov_s = cov_matrix(net(X_syn[idx]))
                    mu_r, cov_r = moments[c]
                    total += float(((mu_r - mu_s) ** 2).sum()
                                   + ((cov_r - cov_s) ** 2).sum())
        return total / (len(self.probes) * self.num_classes)


def main():
    # battleground subset: leverage-wins + reversal + durable-TAME cases;
    # extend to all 18 registry names once the mechanism is confirmed
    DATASETS = ["adult", "electricity", "letter", "magic", "pageblocks",
                "pendigits", "satimage", "shuttle"]
    IPCS = [10, 50]
    K = 10
    CLASSIFIERS = ["rf", "xgboost"]
    BASE_SEED = 1000
    FIXED_INIT_SEED = 777
    RESULTS_DIR = "results_best_of_k"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # (method label, synth_type, embedder, init_seed)
    ARMS = [
        ("random",          "random",         "",         None),
        ("leverage",        "leverage_score", "",         None),
        ("tame",            "tame",           "ln_res_l", None),
        ("tame_fixed_init", "tame",           "ln_res_l", FIXED_INIT_SEED),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []

    for db in DATASETS:
        if db not in DATASET_REGISTRY:
            print(f"SKIP {db}")
            continue
        for ipc in IPCS:
            base_config = {
                "dataset_name": db,
                "device": device,
                "ipc": ipc,
                "dm_iters": 1000,
                "dm_lr": 0.5,
                "dm_batch_real": 128,
                "dm_embedder_size": "base",
                "dm_embed_hidden": 256,
                "dm_embed_dim": embed_dim_for_ipc(ipc),
                "classifier_hidden": [128, 64],
                "classifier_epochs": 20,
                "random_seed": BASE_SEED,
            }
            data0 = prepare_db(base_config, name=db)
            print(f"\n[{db}] ipc={ipc}: building probe scorer ...")
            scorer = ProbeScorer(data0, base_config, device)

            for label, synth_type, embedder, init_seed in ARMS:
                print(f"\n[{db}] ipc={ipc} arm={label} K={K}")
                for k in range(K):
                    run_seed = BASE_SEED + k
                    set_seed(run_seed)

                    config = dict(base_config)
                    config["synth_type"] = synth_type
                    config["dm_embedder_type"] = embedder
                    config["random_seed"] = run_seed  # random/leverage re-seed from this
                    if init_seed is not None:
                        config["init_seed"] = init_seed

                    data = prepare_db(config, name=db)

                    t0 = time.time()
                    try:
                        X_syn, y_syn = synthesize(synth_type=synth_type,
                                                  data=data, config=config)
                    except Exception as e:
                        print(f"  run{k} distillation FAILED: {e}")
                        continue
                    synth_time = time.time() - t0

                    train_data = {
                        "X_train": X_syn, "y_train": y_syn,
                        "X_val": data["X_val"], "y_val": data["y_val"],
                        "input_dim": data["input_dim"],
                        "num_classes": data["num_classes"],
                    }

                    clf_results = {}
                    for clf in CLASSIFIERS:
                        clf_config = dict(config)
                        clf_config["classifier"] = clf
                        try:
                            model = train_classifier(train_data, clf_config)
                            clf_results[clf] = (
                                eval_on_split(model, data, device, "val"),
                                eval_on_split(model, data, device, "test"),
                            )
                        except Exception as e:
                            print(f"  run{k} {clf} FAILED: {e}")

                    # classifier-free set quality (uses fixed probe seeds;
                    # done after training so it can't perturb anything)
                    probe = scorer.score(X_syn, y_syn)

                    for clf, (val_acc, test_acc) in clf_results.items():
                        all_rows.append({
                            "dataset": db, "method": label, "ipc": ipc,
                            "run": k, "seed": run_seed, "classifier": clf,
                            "val_acc": val_acc, "test_acc": test_acc,
                            "probe_score": probe, "synth_time": synth_time,
                        })
                        print(f"  run{k} {clf}: val={val_acc:.4f} "
                              f"test={test_acc:.4f} probe={probe:.4f}")

                    pd.DataFrame(all_rows).to_csv(
                        os.path.join(RESULTS_DIR, "partial.csv"), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/results.csv ({len(df)} rows)")

    analyze(df, RESULTS_DIR)


def analyze(df, results_dir):
    from scipy.stats import spearmanr

    rows = []
    for (db, ipc, clf, method), g in df.groupby(
            ["dataset", "ipc", "classifier", "method"]):
        g = g.dropna(subset=["val_acc", "test_acc"])
        if len(g) < 3:
            continue
        best_val = g.loc[g["val_acc"].idxmax()]
        best_probe = g.loc[g["probe_score"].idxmin()]
        sp_val = spearmanr(g["val_acc"], g["test_acc"]).statistic
        sp_probe = spearmanr(-g["probe_score"], g["test_acc"]).statistic
        rows.append({
            "dataset": db, "ipc": ipc, "classifier": clf, "method": method,
            "mean_test": g["test_acc"].mean(),
            "std_test": g["test_acc"].std(),
            "best_by_val": best_val["test_acc"],
            "best_by_probe": best_probe["test_acc"],
            "oracle": g["test_acc"].max(),
            "gain_val": best_val["test_acc"] - g["test_acc"].mean(),
            "gain_probe": best_probe["test_acc"] - g["test_acc"].mean(),
            "spearman_val_test": sp_val,
            "spearman_probe_test": sp_probe,
            "n_runs": g["run"].nunique(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100)
    print("  PER-GROUP SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("  AGGREGATES BY METHOD (mean over dataset/ipc/classifier groups)")
    print("=" * 100)
    agg = summary.groupby(["ipc", "classifier", "method"])[
        ["mean_test", "std_test", "best_by_val", "best_by_probe", "oracle",
         "gain_val", "gain_probe", "spearman_val_test", "spearman_probe_test"]
    ].mean().round(4)
    print(agg.to_string())


if __name__ == "__main__":
    main()
