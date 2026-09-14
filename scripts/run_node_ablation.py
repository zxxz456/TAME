"""NODE with degenerate (zero) tree init vs randomised tree init, exhaustively.

Supersedes run_node_random_init.py, which only scored the final `best_syn` at a
single IPC and therefore could say whether the fix helps but not when, where or
why. Here every distillation reports its whole trajectory.

TAME never trains the embedder, so NODE's `feature_logits` and `thresholds` stay
at their zero init: softmax(zeros) is uniform, every tree and level reads the same
mean of all coordinates, and the ensemble spans 5 of 24 directions with singular
values at numerical noise (scripts/check_node_degeneracy.py). The paper reads
NODE's poor score as evidence that emulating trees in the embedder does not help;
if the ensemble was inert, that conclusion never had a tree-like embedder behind
it.

The two arms share seeds and differ ONLY in `tree_random_init`, so every
difference below is attributable to it.

Writes, per sweep:
    results.csv    final best_syn scored by each classifier: accuracy, AUC, the
                   selected iteration and its loss, wall-clock
    traces.csv     per iteration: loss, its mean and covariance parts, and the
                   gradient norm the distillation asked for before clipping
    curves.csv     accuracy and AUC of each classifier on the synthetic set every
                   --snap-every iterations, i.e. the learning curve of the data
    per_class.csv  the two loss terms per class, averaged over the second half of
                   the run: which classes the arm fails to match
and per distillation, under <ds>_ipc<k>_<arm>_run<n>/:
    best_syn.pt    the selected set, the raw trace, the per-class array and every
                   snapshot tensor, so any analysis can be redone without
                   distilling again
"""
import os, sys, time, random, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/zxxz6/TAME"); os.chdir("/home/zxxz6/TAME")
from data.prepare_database import prepare_db
from synth.tame_synth import tame_synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+",
                default=["magic", "electricity", "pageblocks", "shuttle"])
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--snap-every", type=int, default=50,
                help="cada cuantas iters evaluar la curva; 0 la apaga. "
                     "Cuesta 3 clasificadores por punto, es lo mas caro del script")
ap.add_argument("--max-embed-dim", type=int, default=256,
                help="tope de p; la covarianza es p x p por clase, asi que sube rapido")
ap.add_argument("--embedder", default="node", choices=["node", "ln_res_l", "dcnv2_base"],
                help="node corre los dos brazos; los demas corren uno solo, "
                     "para comparar contra el NODE arreglado")
ap.add_argument("--arms", nargs="+", default=None, choices=["zero", "random"],
                help="solo aplica a node; por defecto los dos")
ap.add_argument("--logit-std", type=float, default=3.0)
ap.add_argument("--thr-std", type=float, default=1.0)
ap.add_argument("--cpu", action="store_true",
                help="permite correr en CPU; sin esto, no ver la GPU es un error")
ap.add_argument("--redo", action="store_true",
                help="rehace destilaciones que ya tienen best_syn.pt en disco")
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/node_ablation"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
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
# Con node el experimento son dos brazos: tree_random_init apagado y prendido.
# Con cualquier otro embedder esa bandera no existe, asi que hay un solo brazo y se
# etiqueta con el nombre del embedder, para que los dos barridos se puedan
# concatenar en un CSV y comparar NODE arreglado contra LnRes.
if a.embedder == "node":
    _sel = a.arms or ["zero", "random"]
    ARMS = [(n, n == "random") for n in _sel]
else:
    ARMS = [(a.embedder, False)]

# Dimension de salida del embedder. Tiene que quedar por debajo del IPC o la
# covarianza sintetica es deficiente de rango. Son los valores del paper (Sec 4.5).
PAPER_P = {10: 8, 25: 24, 50: 48, 100: 96, 150: 148, 200: 196}

def embed_dim_for(ipc, cap):
    return max(4, min(PAPER_P.get(ipc, max(4, ipc - 4)), cap, ipc - 1))


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

_CACHE = {}

def datos(ds, seed):
    """prepare_db con memoria de una sola entrada.

    La particion depende de la semilla, asi que no se puede sacar del bucle sin
    cambiar el experimento; pero solo hay `runs` particiones distintas y cada una
    se reconstruia arms * ipcs veces. Con `run` como bucle externo, guardar la
    ultima basta y la memoria no crece: para airlines, con 539383 filas, esa
    reconstruccion era la mayor parte del costo de una celda a IPC bajo."""
    k = (ds, seed)
    if k not in _CACHE:
        _CACHE.clear()                     # una sola particion viva a la vez
        _CACHE[k] = prepare_db({"random_seed": seed, "device": DEV}, name=ds)
    return _CACHE[k]


def score(X, y, data, clf, cfg, seed):
    """Train one classifier on a distilled set and read test accuracy and AUC.

    The seed is reset per call so the only thing separating two points of a
    curve is the synthetic set, not the classifier's own initialisation."""
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
    done = set(map(tuple, _d[["dataset", "ipc", "arm", "run"]].drop_duplicates().values))
    print(f"[resume] {len(done)} destilaciones ya registradas en {_res}")

rows, traces, curves, pcs = [], [], [], []
# Run-major: la particion de datos depende de la semilla, asi que recorrer las
# corridas por fuera deja fijo `data` mientras se barren IPC y brazos, y el cache
# de una entrada acierta 2 * len(ipcs) veces seguidas. Cada destilacion hace
# set_seed(seed) justo antes de destilar, asi que el orden de los bucles no
# cambia ningun resultado; las semillas se siguen compartiendo entre brazos y el
# pareo se conserva.
for ds in a.datasets:
  for run in range(a.runs):
    seed = 132 + run
    print(f"\n########## {ds} | corrida {run:02d} (semilla {seed}) | "
          f"{a.embedder} | IPC {a.ipcs} ##########", flush=True)
    for ipc in a.ipcs:
      p_dim = embed_dim_for(ipc, a.max_embed_dim)
      for arm, rnd in ARMS:
            if (ds, ipc, arm, run) in done:
                print(f"[{ds:12s} ipc{ipc:<3d} {arm:6s} run{run:02d}] ya registrada, "
                      f"se salta", flush=True)
                continue
            set_seed(seed)
            t_load = time.time()
            data = datos(ds, seed)            # 0 si la particion ya estaba en cache
            dt_load = time.time() - t_load
            sdir = os.path.join(a.out, f"{ds}_ipc{ipc}_{arm}_run{run:02d}")
            cfg = {
                "device": DEV, "ipc": ipc, "dm_iters": a.iters, "dm_lr": 0.5,
                "dm_batch_real": 128, "dm_embedder_type": a.embedder,
                "dm_embedder_size": "base", "dm_embed_hidden": 256,
                "dm_embed_dim": p_dim,
                "dm_embedder_overrides": {"tree_random_init": rnd,
                                          "logit_std": a.logit_std,
                                          "thr_std": a.thr_std},
                "snapshot_every": a.snap_every, "return_snapshots": bool(a.snap_every),
                "save_dir": sdir, "random_seed": seed,
            }
            # Resume: a distillation already on disk is reused as it stands,
            # snapshots included, rather than repeated. It comes straight out of
            # the .pt, so this is the same result, not an approximation.
            ckpt = os.path.join(sdir, "best_syn.pt")
            if os.path.exists(ckpt) and not a.redo:
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                X_best, y_syn = ck["X_syn"].to(DEV), ck["y_syn"].to(DEV)
                snaps, dt_syn, reused = ck.get("snapshots", []), 0.0, True
            else:
                t0 = time.time()
                out = tame_synthesize(data, cfg)
                X_best, y_syn = out[0], out[1]
                snaps = out[2] if len(out) > 2 else []
                dt_syn, reused = time.time() - t0, False
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)

            # --- trayectoria de la perdida ---
            # Las destilaciones anteriores al cronometro guardaron la traza sin
            # dt_ms. Reusarlas es legitimo, la destilacion no cambio; solo hay que
            # nombrar las columnas por lo que el .pt trae y dejar el tiempo en NaN.
            COLS = ["it", "loss", "loss_mean", "loss_cov", "grad_norm", "dt_ms"]
            tr = pd.DataFrame(ck["trace"], columns=COLS[:len(ck["trace"][0])])
            if "dt_ms" not in tr:
                tr["dt_ms"] = np.nan
            for k, v in [("dataset", ds), ("ipc", ipc), ("arm", arm), ("run", run)]:
                tr.insert(0, k, v)
            append_csv(tr, os.path.join(a.out, "traces.csv"))
            traces.append(tr)

            # --- que clases quedan mal matcheadas, en la segunda mitad ---
            pc = np.asarray(ck["per_class"])                # (iters+1, C, 2)
            half = pc[len(pc) // 2:].mean(0)                # (C, 2)
            pc_df = pd.DataFrame({
                "dataset": ds, "ipc": ipc, "arm": arm, "run": run,
                "clase": np.arange(pc.shape[1]),
                "loss_mean": half[:, 0], "loss_cov": half[:, 1]})
            append_csv(pc_df, os.path.join(a.out, "per_class.csv"))
            pcs.append(pc_df)

            # --- curva de accuracy a lo largo de la destilacion ---
            t1 = time.time()
            cur_curve = []
            for it, X_it in snaps:
                for clf in CLFS:
                    acc, auc = score(X_it.to(DEV), y_syn, data, clf, cfg, seed)
                    cur_curve.append(dict(dataset=ds, ipc=ipc, arm=arm, run=run,
                                          seed=seed, it=int(it), classifier=clf,
                                          acc=acc, auc=auc))
            if cur_curve:
                append_csv(pd.DataFrame(cur_curve), os.path.join(a.out, "curves.csv"))
                curves.extend(cur_curve)
            dt_ev = time.time() - t1

            # --- el best_syn, que es lo que el metodo devuelve de verdad ---
            for clf in CLFS:
                acc, auc = score(X_best, y_syn, data, clf, cfg, seed)
                rows.append(dict(dataset=ds, ipc=ipc, arm=arm, run=run, seed=seed,
                                 classifier=clf, acc=acc, auc=auc,
                                 best_it=int(ck["best_it"]),
                                 best_loss=float(ck["best_loss"]),
                                 grad_norm_mean=float(tr.grad_norm.mean()),
                                 # Tiempos: por destilacion, por iteracion, y el
                                 # costo de reconstruir la particion (0 si venia
                                 # del cache). eval_s es entrenar clasificadores.
                                 synth_s=round(dt_syn, 1), eval_s=round(dt_ev, 1),
                                 load_s=round(dt_load, 1),
                                 ms_per_iter=round(float(tr.dt_ms.mean()), 2),
                                 ms_iter_p50=round(float(tr.dt_ms.median()), 2),
                                 ms_iter_p95=round(float(tr.dt_ms.quantile(0.95)), 2)))
            # NaN en los tres ms_* marca una destilacion reusada de antes del
            # cronometro; el resto de sus columnas es igual de valido.
            append_csv(pd.DataFrame(rows[-3:]), os.path.join(a.out, "results.csv"))
            print(f"[{ds:12s} ipc{ipc:<3d} {arm:6s} run{run:02d}] "
                  f"destilar {'reusada' if reused else f'{dt_syn:6.1f}s'}"
                  f"  curva {len(snaps):2d}pts {dt_ev:6.1f}s  "
                  f"best@{ck['best_it']:4d}  "
                  + "  ".join(f"{r['classifier']}={r['acc']:.4f}" for r in rows[-3:]),
                  flush=True)

# Los CSV ya se escribieron incrementalmente; esto es solo para el resumen.
# El resumen se arma leyendo el CSV, que incluye lo de corridas anteriores.
df = pd.read_csv(_res)

piv = df.pivot_table(index=["dataset", "ipc", "classifier", "run"],
                     columns="arm", values="acc")
piv["delta"] = piv["random"] - piv["zero"]

print("\n=== accuracy por brazo ===")
print(df.pivot_table(index=["dataset", "ipc", "classifier"], columns="arm",
                     values="acc", aggfunc=["mean", "std"]).round(4))

print("\n=== delta (random menos zero), pareado por semilla ===")
print(piv.groupby(["dataset", "ipc", "classifier"]).delta
         .agg(["mean", "median", "std", "min", "max"]).round(4))

print("\n=== delta agregado por IPC y clasificador ===")
print(piv.groupby(["ipc", "classifier"]).delta
         .agg(media="mean", mediana="median", gana=lambda x: int((x > 0).sum()),
              n="count").round(4))

print("\nCSV ->", a.out)
