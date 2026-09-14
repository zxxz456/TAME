"""Contaminated vs clean best-loss selection for the adversarial TAME variant.

DAIRY 1: the scalar that picks `best_syn` in `tame_synth_critic.py` includes the
WGAN critic's score. That score is unbounded (the gradient penalty bounds the norm
of the critic's gradient, not its value) and drifts while the critic trains, so the
checkpoint could be chosen for reasons unrelated to the synthetic data.

Both candidate iterates come out of a **single** distillation: the selection rule
only decides which one is returned, so `tame_critic_synthesize` now tracks the
argmin of `loss_dm` and of `loss_total` side by side. Running two separate arms
would have been weaker, because each arm sees a different embedder sequence (the
sampler seeds itself from the wall clock) and part of the difference would be the
path rather than the rule.

Writes, per sweep:
    results.csv    one row per (dataset, ipc, run, classifier): accuracy and AUC
                   under each rule, their paired delta, the iteration each rule
                   picked, and the mean gradient weight of the critic term
    traces.csv     per iteration: loss_dm, loss_total, their gap, the mean and
                   covariance terms, and the gradient norm of each term
and per distillation:
    <run>/snapshots.npz   the synthetic set every --snap-every iterations
    <run>/best_syn.pt     both selected iterates plus the raw trace
"""
import os, sys, time, random, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/zxxz6/TAME"); os.chdir("/home/zxxz6/TAME")
from data.prepare_database import prepare_db
from synth.registry import synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier

ap = argparse.ArgumentParser()
# Por defecto los cinco datasets pedidos y los tres presupuestos de la Tabla 11.
ap.add_argument("--datasets", nargs="+",
                default=["magic", "letter", "airlines", "adult", "madelon"])
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--max-embed-dim", type=int, default=256,
                help="tope de p; la covarianza es p x p, asi que sube rapido")
ap.add_argument("--embedder", default="ln_res_l",
                choices=["ln_res_l", "dcnv2_base", "node"])
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--grad-every", type=int, default=10,
                help="cada cuantas iters medir las normas de gradiente de los dos terminos")
ap.add_argument("--snap-every", type=int, default=25,
                help="cada cuantas iters guardar el conjunto sintetico completo")
ap.add_argument("--cpu", action="store_true",
                help="permite correr en CPU; sin esto, no ver la GPU es un error")
ap.add_argument("--redo", action="store_true",
                help="rehace destilaciones que ya tienen best_syn.pt en disco")
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/critic_clean"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Una caida silenciosa a CPU cuesta dias: PyTorch no falla, solo se pone 200 veces
# mas lento. Paso exactamente eso cuando el servidor MPS quedo en estado fatal
# tras matar clientes con SIGKILL. Si hay GPU en la maquina y torch no la ve, es
# un problema del entorno y hay que verlo antes de arrancar, no a las 10 horas.
if DEV == "cpu" and not a.cpu:
    import shutil, subprocess
    hay_gpu = shutil.which("nvidia-smi") and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True).returncode == 0
    raise SystemExit(
        "torch no ve la GPU" + (" pero nvidia-smi si la lista: revisa el entorno "
        "(MPS atorado? CUDA_VISIBLE_DEVICES?). Usa --cpu si de verdad la quieres."
        if hay_gpu else ". Usa --cpu si es lo que quieres."))
print(f"[dispositivo] {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else ""))
CLFS = ["mlp", "rf", "xgboost"]

# Dimension de salida del embedder. Tiene que quedar por debajo del IPC o la
# covarianza sintetica es deficiente de rango. Los valores del paper (Sec. 4.5)
# para IPC 10..200, y de ahi en adelante ipc-4 topado, porque el termino de
# covarianza cuesta p x p por clase.
PAPER_P = {10: 8, 25: 24, 50: 48, 100: 96, 150: 148, 200: 196}

def embed_dim_for(ipc, cap):
    p = PAPER_P.get(ipc, max(4, ipc - 4))
    return max(4, min(p, cap, ipc - 1))


def append_csv(df, path):
    """Append rows to a CSV, writing the header only the first time.

    The sweeps run for hours; dumping every table at the end means a crash or a
    kill in hour thirteen loses thirteen hours of finished work. Each process
    owns its own --out, so no two of them ever write the same file.

    The schema can grow between runs: a resumed sweep may add columns that the
    file on disk does not have. Appending wider rows under a narrower header
    silently corrupts the file, so a mismatch is reconciled once by rewriting
    with the union of both column sets; afterwards the header matches and the
    append is a plain write again."""
    if not os.path.exists(path):
        df.to_csv(path, index=False)
        return
    with open(path) as f:
        viejo = f.readline().strip().split(",")
    if viejo == list(df.columns):
        df.to_csv(path, mode="a", header=False, index=False)
        return
    prev = pd.read_csv(path)
    pd.concat([prev, df], ignore_index=True).to_csv(path, index=False)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def score(X, y, data, clf, cfg, seed):
    """Train one classifier on a distilled set and read test accuracy."""
    set_seed(seed)
    td = {"X_train": X, "y_train": y, "X_val": data["X_val"], "y_val": data["y_val"],
          "input_dim": data["input_dim"], "num_classes": data["num_classes"]}
    m = train_classifier(td, {**cfg, "classifier": clf,
                              "classifier_hidden": [128, 64], "classifier_epochs": 20})
    acc, auc = evaluate_classifier(m, data, DEV)
    return float(acc), float(auc)

# Combinations already present in results.csv, so relaunching after a kill
# neither recomputes nor duplicates them. --redo ignores this.
done = set()
_res = os.path.join(a.out, "results.csv")
if os.path.exists(_res) and not a.redo:
    _d = pd.read_csv(_res)
    done = set(map(tuple, _d[["dataset", "ipc", "run"]].drop_duplicates().values))
    print(f"[resume] {len(done)} destilaciones ya registradas en {_res}")

rows, traces = [], []
for ds in a.datasets:
  for ipc in a.ipcs:
    p_dim = embed_dim_for(ipc, a.max_embed_dim)
    print(f"\n===== {ds} | IPC={ipc} | p={p_dim} | embedder {a.embedder} =====",
          flush=True)
    for run in range(a.runs):
        if (ds, ipc, run) in done:
            print(f"  run{run:02d}  ya registrada, se salta", flush=True)
            continue
        seed = 132 + run
        set_seed(seed)
        data = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
        sdir = os.path.join(a.out, f"{ds}_ipc{ipc}_run{run:02d}")
        cfg = {
            "device": DEV, "synth_type": "tame_critic", "ipc": ipc,
            "dm_iters": a.iters, "dm_lr": 0.5, "dm_batch_real": 128,
            "dm_embedder_type": a.embedder, "dm_embedder_size": "base",
            "dm_embed_hidden": 256, "dm_embed_dim": embed_dim_for(ipc, a.max_embed_dim),
            "dm_use_critic": True, "dm_adv_weight": 0.05, "dm_n_critic": 3,
            "grad_log_every": a.grad_every, "snapshot_every": a.snap_every,
            "save_dir": sdir, "random_seed": seed,
        }
        # Reanudar: una destilacion ya guardada se reusa tal cual en vez de
        # repetirse. Los tensores salen del .pt, asi que el resultado es
        # identico, no una aproximacion.
        ckpt = os.path.join(sdir, "best_syn.pt")
        if os.path.exists(ckpt) and not a.redo:
            dt, reused = 0.0, True
        else:
            t0 = time.time()
            _, y_syn = synthesize("tame_critic", data, cfg)
            dt, reused = time.time() - t0, False

        # Both iterates and the trace come out of that one distillation.
        ck = torch.load(ckpt, map_location=DEV, weights_only=False)
        if reused:
            y_syn = ck["y_syn"].to(DEV)
        X_dm,  it_dm  = ck["X_syn_dm"].to(DEV),    int(ck["best_dm_it"])
        X_tot, it_tot = ck["X_syn_total"].to(DEV), int(ck["best_total_it"])

        # Igual que en run_node_ablation: las destilaciones anteriores al
        # cronometro traen la traza sin las dos columnas de tiempo.
        COLS = ["it", "loss_dm", "loss_total", "gap", "loss_mean", "loss_cov",
                "g_dm", "g_adv", "dt_ms", "dt_critic_ms"]
        tr = pd.DataFrame(ck["trace"], columns=COLS[:len(ck["trace"][0])])
        for c in ("dt_ms", "dt_critic_ms"):
            if c not in tr:
                tr[c] = np.nan
        tr["g_ratio"] = tr.g_adv / tr.g_dm
        tr["dt_syn_ms"] = tr.dt_ms - tr.dt_critic_ms     # la mitad que mueve los datos
        tr.insert(0, "run", run); tr.insert(0, "ipc", ipc); tr.insert(0, "dataset", ds)
        append_csv(tr, os.path.join(a.out, "traces.csv"))
        traces.append(tr)

        # La trayectoria completa del conjunto sintetico, un array por iteracion
        # muestreada. Pesa poco y permite rehacer cualquier analisis sin volver
        # a destilar.
        snaps = ck.get("snapshots", [])
        if snaps:
            np.savez_compressed(
                os.path.join(sdir, "snapshots.npz"),
                iters=np.array([i for i, _ in snaps]),
                X=np.stack([x.cpu().numpy() for _, x in snaps]),
                y=y_syn.detach().cpu().numpy())

        for clf in CLFS:
            acc_dm,  auc_dm  = score(X_dm,  y_syn, data, clf, cfg, seed)
            acc_tot, auc_tot = score(X_tot, y_syn, data, clf, cfg, seed)
            rows.append(dict(dataset=ds, ipc=ipc, run=run, seed=seed, classifier=clf,
                             acc_clean=acc_dm, acc_contaminated=acc_tot,
                             delta=acc_dm - acc_tot,
                             auc_clean=auc_dm, auc_contaminated=auc_tot,
                             it_clean=it_dm, it_contaminated=it_tot,
                             it_gap=it_dm - it_tot,
                             same_iterate=int(it_dm == it_tot),
                             loss_dm=float(ck["best_dm_loss"]),
                             loss_total=float(ck["best_total_loss"]),
                             g_dm_mean=float(tr.g_dm.mean(skipna=True)),
                             g_adv_mean=float(tr.g_adv.mean(skipna=True)),
                             g_ratio_mean=float(tr.g_ratio.mean(skipna=True)),
                             # Tiempos: por destilacion y por iteracion, con el
                             # bloque del critico separado del paso sintetico.
                             synth_s=round(dt, 1),
                             ms_per_iter=round(float(tr.dt_ms.mean()), 2),
                             ms_critic=round(float(tr.dt_critic_ms.mean()), 2),
                             ms_syn=round(float(tr.dt_syn_ms.mean()), 2)))
        last = rows[-3:]
        append_csv(pd.DataFrame(last), os.path.join(a.out, "results.csv"))
        print(f"  run{run:02d}  {'reusada' if reused else f'{dt:5.1f}s'}  it limpio={it_dm:4d} contaminado={it_tot:4d}"
              + ("  (mismo iterado)" if it_dm == it_tot else "")
              + "  " + "  ".join(f"{r['classifier']}: {r['delta']:+.4f}" for r in last),
              flush=True)

# Los CSV ya se escribieron incrementalmente; esto es solo para el resumen.
# El resumen se arma leyendo los CSV, que incluyen lo de corridas anteriores.
df = pd.read_csv(_res)
tf = pd.read_csv(os.path.join(a.out, "traces.csv"))

one = df.drop_duplicates(["dataset", "ipc", "run"])
print(f"\n=== las dos reglas eligieron el MISMO iterado en "
      f"{one.same_iterate.sum()} de {len(one)} destilaciones ===")
print(one.groupby(["dataset","ipc"]).same_iterate.agg(["sum","count"]))

print("\n=== iteracion elegida por cada regla ===")
print(one.groupby(["dataset","ipc"])[["it_clean","it_contaminated","it_gap"]]
        .agg(["mean","median"]).round(1))

print("\n=== delta de accuracy (limpio menos contaminado), pareado ===")
print(df.groupby(["dataset","ipc","classifier"]).delta
        .agg(["mean","median","std","min","max"]).round(4))

print("\n=== gradiente: cuanto pesa el termino adversarial frente al DM ===")
print(tf.groupby(["dataset","ipc"])[["g_dm","g_adv","g_ratio"]]
        .agg(["mean","median","max"]).round(5))

print("\n=== separacion entre los dos escalares ===")
print(tf.groupby(["dataset","ipc"])[["loss_dm","loss_total","gap"]]
        .agg(["mean","min","max"]).round(4))

print("\nCSV ->", os.path.join(a.out, "results.csv"))
print("CSV ->", os.path.join(a.out, "traces.csv"))
