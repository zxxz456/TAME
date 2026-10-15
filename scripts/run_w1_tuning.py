"""Pick the weight of the W1 marginal term on validation, before comparing it
against plain TAME.

The term (``dm_w1_weight`` in synth/tame_synth.py) adds to Eq. 7 the 1-D
Wasserstein distance between each synthetic column and the real one, per class,
in data space. Running the comparison at whatever weight happened to be the
default would test the default, not the term, so the weight is chosen first.

**The grid comes from a calibration**, ln_res_l, IPC 50, seed 1000, on the ten
datasets below. The norm of the W1 gradient at weight 1 is
sqrt(C * ipc * d) / ipc, every entry getting 1/ipc, and against the median norm
of the moment gradient it gives a ratio between 0.36 (magic) and 4.5
(madelon), about 1 on most. So weight 1 is where the two terms push about
equally. At weight 10 the total gradient exceeds the clip of 10 on every
iteration of spambase, madelon and letter, so the moments get scaled down with
it: that is the "the term dominates" end. The grid, 0.1 to 10, spans from
barely felt to dominant. Plain TAME, weight 0, runs alongside as `none`, so the
tuning also says whether any weight beats it on validation.

Same calibration, on the distance itself: plain TAME moves the marginals AWAY
from the real ones on spambase (W1 summed over columns, per class, 6.7 at the
init to 11.5), credit, higgs and letter, and closer on magic, pendigits and
satimage. With the term at weight 1, spambase ends at 1.1 and madelon goes from
84 to 3.

**The choice is made on validation, never on test**, with seeds 1000-1009,
disjoint from the 132-141 of the comparison, exactly as the one-hot tuning did.

Ten datasets, the two null blocks of the one-hot ablation: binary and
multi-class with no categorical columns, so the term acts on every column.

    THREADS=3 bash scripts/run_sharded.sh run_w1_tuning.py ~/tame_runs/w1_tune 5 \\
      magic spambase credit madelon higgs letter pendigits satimage segment pageblocks
"""
import os, sys, argparse
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/zxxz6/TAME"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/zxxz6/TAME")
from w1_common import BIN_SIN, MULTI_SIN, correr, linea, reanudar

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=BIN_SIN + MULTI_SIN)
ap.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0])
ap.add_argument("--ipcs", nargs="+", type=int, default=[50])
ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1010)),
                help="disjoint from the comparison's 132-141 on purpose")
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--embedders", nargs="+", default=["ln_res_l", "dcnv2_base"])
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/w1_tune"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
BRAZOS = ["none"] + [f"w1@{w:g}" for w in a.weights]

csv = os.path.join(a.out, "results.csv")
rows, hecho = reanudar(csv, ["dataset", "embedder", "ipc", "brazo", "seed"])
for ds in a.datasets:
    print(f"##### {ds} | {len(BRAZOS)} arms -> {BRAZOS}", flush=True)
    for emb in a.embedders:
        for ipc in a.ipcs:
            for etq in BRAZOS:
                for seed in a.seeds:
                    if (ds, emb, ipc, etq, seed) in hecho:
                        continue
                    nuevas = correr(ds, emb, ipc, etq, seed, a.iters)
                    rows += nuevas
                    print(linea(nuevas), flush=True)
                    pd.DataFrame(rows).to_csv(csv, index=False)

df = pd.DataFrame(rows)
df.to_csv(csv, index=False)
pd.set_option("display.width", 240)

# The ranking that picks the weight: validation, trees, per block and embedder.
# Test sits next to it only to show whether the two orderings agree.
ARB = ["rf", "xgboost"]
t = df[df.classifier.isin(ARB)]
for bl in sorted(t.bloque.unique()):
    for emb in sorted(t.embedder.unique()):
        s = (t[(t.bloque == bl) & (t.embedder == emb)].groupby("brazo")
             [["val_bestloss", "acc_bestloss", "val_final", "acc_final",
               "w1_bestloss", "sw_bestloss", "best_en_init"]].mean())
        print(f"\n=== {bl} | {emb} | trees, mean over datasets and seeds ===")
        print(s.sort_values("val_bestloss", ascending=False).round(4).to_string())
print("\n=== mlp, validation ===")
print(df[df.classifier == "mlp"].pivot_table(index="brazo", columns=["bloque", "embedder"],
                                             values="val_bestloss").round(4).to_string())

# Each weight against plain TAME, paired on dataset, embedder, seed and
# classifier, on validation. What decides is the trees row.
base = df[df.brazo == "none"].set_index(["dataset", "embedder", "ipc", "seed", "classifier"])
filas = []
for etq in BRAZOS[1:]:
    s = df[df.brazo == etq].set_index(base.index.names)
    for grupo, clfs in (("trees", ARB), ("mlp", ["mlp"])):
        m = s.index.get_level_values("classifier").isin(clfs)
        d = (s[m].val_bestloss - base.val_bestloss).dropna()
        if len(d):
            filas.append(dict(brazo=etq, clasificador=grupo, n=len(d), delta_val=d.mean(),
                              gana=int((d > 0).sum()),
                              wilcoxon=stats.wilcoxon(d)[1] if np.ptp(d) > 0 else np.nan))
print("\n=== each weight minus none, validation, all datasets ===")
print(pd.DataFrame(filas).round(4).to_string(index=False))
v = t[t.brazo != "none"].groupby("brazo").val_bestloss.mean()
if len(v):
    print(f"\nchosen by validation (trees, every block and embedder): {v.idxmax()}")
print(f"\nCSV -> {csv}")
