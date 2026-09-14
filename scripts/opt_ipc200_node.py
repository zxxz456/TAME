"""The published core against the optimised one, at the largest budget, with NODE.

The earlier equivalence run covered three embedders and three budgets on two
datasets, and turned up something worth a closer look: the advantage of the
optimised core shrinks as the budget grows, and with NODE at IPC 200 on a
five-class dataset it disappeared altogether (0.97x, the optimised path a hair
slower). That was a single cell, on a single dataset, with five seeds.

This widens exactly that cell. Only IPC 200, only NODE, but over five binary and
five multiclass datasets with five seeds each, so the question "does the batched
path stop paying at the largest budget, and does that depend on the number of
classes" gets an answer with a spread attached instead of one number.

Both versions run back to back on the same seed, so whatever the machine is doing
hits the pair equally. Everything is sequential on purpose: two distillations
sharing the device contend for it, and the contention does not fall on the two
versions evenly (the published core allocates a fresh embedder every iteration,
and cudaMalloc serialises across processes). A ratio measured under load is not
the ratio.

Measured per distillation:

  tiempo     total seconds, and milliseconds per iteration from a CUDA event
             recorded at the top of every iteration, so the spread is there too
  memoria    peak device memory, which is per-process and so immune to neighbours
  forma      moments and effective rank of the distilled set, to confirm the two
             versions still agree at this budget

Writes ~/tame_runs/opt/ipc200/runs.csv (one row per distillation) and
.../iters.csv (one row per iteration). Resumable: a cell already in runs.csv is
skipped unless --redo.
"""
import argparse
import csv
import importlib.util
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, "/home/zxxz6/TAME")
os.chdir("/home/zxxz6/TAME")
from data.prepare_database import prepare_db

ap = argparse.ArgumentParser()
# Cinco y cinco. Los binarios cubren dimensiones de 10 a 68, los multiclase
# cubren C de 5 a 10, que es la variable de la que depende el camino viejo.
ap.add_argument("--binarios", nargs="+",
                default=["magic", "electricity", "spambase", "phishing", "german"])
ap.add_argument("--multiclase", nargs="+",
                default=["pageblocks", "satimage", "segment", "shuttle", "pendigits"])
ap.add_argument("--ipc", type=int, default=200)
ap.add_argument("--embedder", default="node")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--commit", default="ac15fc1",
                help="commit con el nucleo previo a la optimizacion")
ap.add_argument("--redo", action="store_true", help="rehace celdas ya escritas")
ap.add_argument("--dry-run", action="store_true", help="solo imprime la rejilla")
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/opt/ipc200"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    raise SystemExit("hace falta cuda: medir tiempos en cpu no dice nada del cambio")
PAPER_P = {10: 8, 25: 24, 50: 48, 100: 96, 150: 148, 200: 196}
CSV_RUNS = f"{a.out}/runs.csv"
CSV_ITERS = f"{a.out}/iters.csv"


def p_de(ipc):
    """La dimension del embedding que el barrido usa para ese presupuesto."""
    return max(4, min(PAPER_P.get(ipc, max(4, ipc - 4)), 256, ipc - 1))


def _estado_maquina():
    """Condiciones en las que se tomo la medida.

    El cociente entre las dos versiones se infla bajo contencion y no lo hace de
    forma simetrica, asi que sin este registro una medida con la maquina llena no
    se distingue de una con la maquina vacia."""
    try:
        n_gpu = len([l for l in subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.splitlines() if l.strip()])
    except Exception:
        n_gpu = -1
    la = os.getloadavg()[0]
    return dict(carga_cpu=round(la, 2), nucleos=os.cpu_count(),
                procesos_gpu=n_gpu, maquina_libre=int(n_gpu <= 1 and la < 2))


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# --- la version previa, sacada del commit ---
TMP = tempfile.mkdtemp()
src = subprocess.run(["git", "show", f"{a.commit}:synth/tame_synth.py"],
                     capture_output=True, text=True).stdout
if not src.strip():
    raise SystemExit(f"no pude leer synth/tame_synth.py de {a.commit}")
with open(f"{TMP}/viejo.py", "w") as f:
    f.write(src)
spec = importlib.util.spec_from_file_location("tame_viejo", f"{TMP}/viejo.py")
VIEJO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VIEJO)
import synth.tame_synth as NUEVO
VERSIONES = [("viejo", VIEJO), ("nuevo", NUEVO)]


class Reloj:
    """Un evento de CUDA al inicio de cada iteracion.

    Registrar un evento no sincroniza: se encola como cualquier kernel y su
    marca de tiempo se lee una sola vez, al final. Medir con perf_counter daria
    el tiempo de la CPU encolando, que en la version nueva corre por delante de
    la GPU y no es el tiempo de la iteracion."""

    def __init__(self, n):
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(n + 2)]
        self.i = 0

    def marca(self):
        if self.i < len(self.ev):
            self.ev[self.i].record()
            self.i += 1

    def ms(self):
        torch.cuda.synchronize()            # la unica sincronizacion, y va al final
        return [self.ev[k].elapsed_time(self.ev[k + 1]) for k in range(self.i - 1)]


def engancha(mod, reloj):
    """Marca cada iteracion sin editar el codigo de ninguna de las dos versiones.

    Cada version llama exactamente una funcion por iteracion y por vista: la
    vieja construye el embedder con sample_random_embedder, la nueva redibuja sus
    pesos con reinit_embedder_. Envolver esa llamada da el mismo punto logico en
    las dos.

    Devuelve la funcion que deshace el enganche."""
    orig_sample = mod.sample_random_embedder
    tiene_reinit = hasattr(mod, "reinit_embedder_")

    def sample(*args, **kw):
        if not tiene_reinit:                # la vieja construye una por iteracion
            reloj.marca()
        return orig_sample(*args, **kw)

    mod.sample_random_embedder = sample
    orig_reinit = None
    if tiene_reinit:
        orig_reinit = mod.reinit_embedder_

        def reinit(net):
            reloj.marca()                   # la nueva redibuja una por iteracion
            return orig_reinit(net)

        mod.reinit_embedder_ = reinit

    def suelta():
        mod.sample_random_embedder = orig_sample
        if orig_reinit is not None:
            mod.reinit_embedder_ = orig_reinit

    return suelta


def resumen_tiempos(ms):
    """Estadisticos de la serie por iteracion.

    La primera iteracion y la ultima se quitan: la primera carga la compilacion
    de kernels y el calentamiento del asignador, y la ultima cierra sobre el
    trabajo posterior al bucle."""
    v = np.asarray(ms[1:-1]) if len(ms) > 2 else np.asarray(ms)
    if not len(v):
        return dict(ms_media=float("nan"), ms_mediana=float("nan"),
                    ms_std=float("nan"), ms_p10=float("nan"),
                    ms_p90=float("nan"), ms_max=float("nan"), n_iters_medidas=0)
    return dict(ms_media=round(float(v.mean()), 4),
                ms_mediana=round(float(np.median(v)), 4),
                ms_std=round(float(v.std()), 4),
                ms_p10=round(float(np.percentile(v, 10)), 4),
                ms_p90=round(float(np.percentile(v, 90)), 4),
                ms_max=round(float(v.max()), 4),
                n_iters_medidas=int(len(v)))


def forma(X):
    """Lo minimo para confirmar que las dos versiones siguen coincidiendo."""
    x = X.detach().float()
    return dict(x_media=round(float(x.mean()), 6), x_std=round(float(x.std()), 6),
                x_min=round(float(x.min()), 4), x_max=round(float(x.max()), 4),
                rango=int(torch.linalg.matrix_rank(x - x.mean(0)).item()))


def append_csv(ruta, fila):
    """Escribe una fila y crea el encabezado si el archivo aun no existe.

    Una fila por destilacion, escrita en cuanto termina: la corrida completa son
    horas y no tiene sentido perderla toda si algo la interrumpe."""
    nuevo = not os.path.exists(ruta) or os.path.getsize(ruta) == 0
    with open(ruta, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def hechas():
    """Celdas ya escritas, para poder retomar una corrida interrumpida."""
    if a.redo or not os.path.exists(CSV_RUNS):
        return set()
    with open(CSV_RUNS) as f:
        return {(r["dataset"], r["version"], int(r["seed"]))
                for r in csv.DictReader(f) if r.get("dataset")}


# --- la rejilla ---
DATASETS = ([(d, "binario") for d in a.binarios] +
            [(d, "multiclase") for d in a.multiclase])
YA = hechas()
P = p_de(a.ipc)
total = len(DATASETS) * a.seeds * 2
pend = [(ds, tipo, s, ver) for ds, tipo in DATASETS for s in range(a.seeds)
        for ver, _ in VERSIONES if (ds, ver, 400 + s) not in YA]

print(f"[versiones] viejo desde {a.commit}  |  nuevo desde el arbol de trabajo")
print(f"[dispositivo] {DEV} ({torch.cuda.get_device_name(0)})")
print(f"[rejilla] ipc={a.ipc} p={P} | embedder={a.embedder} | "
      f"{a.iters} iters | {a.seeds} semillas")
print(f"[rejilla] {len(a.binarios)} binarios + {len(a.multiclase)} multiclase "
      f"x {a.seeds} semillas x 2 versiones = {total} destilaciones")
print(f"[rejilla] {len(YA)} ya hechas, {len(pend)} pendientes, en serie")
if a.dry_run:
    for ds, tipo, s, ver in pend:
        print(f"  {ds:12s} {tipo:10s} semilla {400+s} {ver}")
    raise SystemExit(0)

t_inicio = time.perf_counter()
n_hechas = 0
for ds, tipo in DATASETS:
    if all((ds, ver, 400 + s) in YA for s in range(a.seeds) for ver, _ in VERSIONES):
        print(f"\n########## {ds} | completo, se salta ##########", flush=True)
        continue
    data = prepare_db({"random_seed": 132, "device": DEV}, name=ds)
    C, d = int(data["num_classes"]), int(data["input_dim"])
    filas_min = C * a.ipc + C * 128          # lo que cruza el embedder por iteracion
    print(f"\n########## {ds} | {tipo} | C={C} d={d} | "
          f"{filas_min} filas por iteracion ##########", flush=True)

    for s in range(a.seeds):
        seed = 400 + s
        # Las dos versiones seguidas sobre la misma semilla: lo que la maquina
        # este haciendo le toca igual a las dos.
        for ver, mod in VERSIONES:
            if (ds, ver, seed) in YA:
                continue
            cfg = dict(device=DEV, ipc=a.ipc, dm_iters=a.iters, dm_lr=0.5,
                       dm_batch_real=128, dm_embedder_type=a.embedder,
                       dm_embedder_size="base", dm_embed_hidden=256,
                       dm_embed_dim=P, random_seed=seed)
            reloj = Reloj(a.iters + 1)
            suelta = engancha(mod, reloj)
            set_seed(seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                X, y = mod.tame_synthesize(data, cfg)[:2]
            finally:
                suelta()
            reloj.marca()                    # cierra la ultima iteracion
            ms = reloj.ms()
            dt = time.perf_counter() - t0
            mem = torch.cuda.max_memory_allocated() / 2 ** 20

            fila = dict(version=ver, dataset=ds, tipo=tipo, C=C, d=d,
                        embedder=a.embedder, ipc=a.ipc, p=P,
                        seed=seed, iters=a.iters,
                        total_s=round(dt, 3),
                        ms_por_iter=round(dt / (a.iters + 1) * 1000, 4),
                        setup_ms=round(dt * 1000 - float(np.sum(ms)), 2),
                        **resumen_tiempos(ms),
                        pico_mib=round(mem, 1),
                        filas_por_iter=filas_min, filas_syn=int(X.shape[0]),
                        **forma(X), **_estado_maquina())
            append_csv(CSV_RUNS, fila)
            with open(CSV_ITERS, "a", newline="") as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(["dataset", "tipo", "C", "version", "seed", "it", "ms"])
                for i, v in enumerate(ms):
                    w.writerow([ds, tipo, C, ver, seed, i, round(v, 4)])

            n_hechas += 1
            transcurrido = time.perf_counter() - t_inicio
            falta = transcurrido / n_hechas * (len(pend) - n_hechas)
            print(f"  {ds:12s} semilla {seed} {ver:6s}  "
                  f"{dt:7.1f}s   {fila['ms_mediana']:7.2f} ms/iter "
                  f"(p10 {fila['ms_p10']:6.2f} p90 {fila['ms_p90']:6.2f})   "
                  f"{mem:7.1f} MiB   |  {n_hechas}/{len(pend)}  "
                  f"faltan {falta/60:.0f} min", flush=True)

        # El cociente del par, en cuanto las dos versiones de esta semilla estan.
        with open(CSV_RUNS) as f:
            r = [x for x in csv.DictReader(f)
                 if x["dataset"] == ds and int(x["seed"]) == seed]
        if len(r) == 2:
            v = {x["version"]: x for x in r}
            fv, fn = float(v["viejo"]["total_s"]), float(v["nuevo"]["total_s"])
            print(f"  {'':12s} semilla {seed} par     "
                  f"viejo {fv:7.1f}s  nuevo {fn:7.1f}s  -> {fv/max(fn,1e-9):5.2f}x   "
                  f"std(X) {float(v['viejo']['x_std']):.4f} vs "
                  f"{float(v['nuevo']['x_std']):.4f}", flush=True)

print(f"\nCSV -> {CSV_RUNS}")
print(f"CSV -> {CSV_ITERS}")
