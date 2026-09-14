"""The published core against the optimised one, over any grid.

`opt_ipc200_node.py` asked one question with a fixed grid: NODE at IPC 200, where
the advantage of the batched path had been seen to vanish. This is the same
measurement with nothing fixed. Datasets, embedders, budgets and the number of
seeds to average over are all options, so the same comparison can be pointed at
whichever cell is in question.

The reason to want the whole grid rather than the time alone is peak memory. The
two changes that buy the speed also move where the work sits: stacking the classes
into a leading dimension puts C times more rows through the embedder at once, so
the activations the backward pass has to keep grow by the same factor. That is the
trade, and it is the candidate explanation for the advantage shrinking as the
budget grows. Time alone cannot show it; peak memory can, and it is measured per
process, so a neighbour on the device does not contaminate it.

For a memory sweep alone there is no need to pay for a full distillation: the peak
is reached within the first few iterations and does not grow after that, so
`--iters 20` gives the same figure in minutes instead of hours.

Measured per distillation, exactly as in the IPC 200 script and with the same CSV
schema, so the two sets of results concatenate:

  tiempo     total seconds, and milliseconds per iteration from a CUDA event
             recorded at the top of every iteration, so the spread is there too
  memoria    peak device memory of this process
  forma      moments and effective rank of the distilled set, to confirm the two
             versions still agree

Writes ~/tame_runs/opt/versiones/runs.csv and .../iters.csv. Resumable: a cell
already in runs.csv is skipped unless --redo.
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
ap.add_argument("--datasets", nargs="+",
                default=["magic", "electricity", "spambase", "phishing", "german",
                         "pageblocks", "satimage", "segment", "shuttle", "pendigits"],
                help="el tipo (binario o multiclase) sale del propio dataset")
ap.add_argument("--embedders", nargs="+",
                default=["ln_res_l", "dcnv2_base", "node"])
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--seeds", type=int, default=3,
                help="repeticiones por celda, para promediar")
ap.add_argument("--iters", type=int, default=1000,
                help="para medir solo memoria bastan ~20: el pico se alcanza "
                     "en las primeras iteraciones y ya no crece")
ap.add_argument("--versiones", nargs="+", default=["viejo", "nuevo"],
                choices=["viejo", "nuevo"])
ap.add_argument("--commit", default="ac15fc1",
                help="commit con el nucleo previo a la optimizacion")
ap.add_argument("--redo", action="store_true", help="rehace celdas ya escritas")
ap.add_argument("--dry-run", action="store_true", help="solo imprime la rejilla")
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/opt/versiones"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    raise SystemExit("hace falta cuda: medir tiempos y memoria en cpu no dice "
                     "nada del cambio")
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
    se distingue de una con la maquina vacia. La memoria pico no sufre de esto:
    max_memory_allocated es por proceso."""
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
MODULOS = {"viejo": VIEJO, "nuevo": NUEVO}
VERSIONES = [(v, MODULOS[v]) for v in a.versiones]


class Reloj:
    """Un evento de CUDA al inicio de cada iteracion.

    Registrar un evento no sincroniza: se encola como cualquier kernel y su marca
    de tiempo se lee una sola vez, al final. Medir con perf_counter daria el
    tiempo de la CPU encolando, que en la version nueva corre por delante de la
    GPU y no es el tiempo de la iteracion."""

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

    Una fila por destilacion, escrita en cuanto termina: la rejilla completa son
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
        return {(r["dataset"], r["embedder"], int(r["ipc"]), r["version"],
                 int(r["seed"])) for r in csv.DictReader(f) if r.get("dataset")}


# --- la rejilla ---
YA = hechas()
CELDAS = [(ds, emb, ipc, s, ver)
          for ds in a.datasets for emb in a.embedders for ipc in a.ipcs
          for s in range(a.seeds) for ver, _ in VERSIONES]
pend = [c for c in CELDAS if (c[0], c[1], c[2], c[4], 400 + c[3]) not in YA]

print(f"[versiones] viejo desde {a.commit}  |  nuevo desde el arbol de trabajo")
print(f"[dispositivo] {DEV} ({torch.cuda.get_device_name(0)})")
print(f"[rejilla] {len(a.datasets)} datasets x {len(a.embedders)} embedders x "
      f"{len(a.ipcs)} ipcs x {a.seeds} semillas x {len(VERSIONES)} versiones "
      f"= {len(CELDAS)} destilaciones")
print(f"[rejilla] embedders {' '.join(a.embedders)} | "
      f"ipcs {' '.join(str(i) for i in a.ipcs)} | {a.iters} iters")
print(f"[rejilla] {len(CELDAS) - len(pend)} ya hechas, {len(pend)} pendientes, "
      f"en serie")
if a.iters < 100:
    print(f"[aviso] con {a.iters} iters el tiempo por destilacion no es "
          f"comparable con el de una corrida real; la memoria pico si lo es")
if a.dry_run:
    for ds, emb, ipc, s, ver in pend:
        print(f"  {ds:12s} {emb:11s} ipc{ipc:<4d} semilla {400+s} {ver}")
    raise SystemExit(0)

t_inicio = time.perf_counter()
n_hechas = 0
for ds in a.datasets:
    if all((ds, e, i, v, 400 + s) in YA for e in a.embedders for i in a.ipcs
           for s in range(a.seeds) for v, _ in VERSIONES):
        print(f"\n########## {ds} | completo, se salta ##########", flush=True)
        continue
    try:
        # Una sola carga por dataset; el resto de la rejilla se mueve por dentro.
        data = prepare_db({"random_seed": 132, "device": DEV}, name=ds)
    except Exception as e:
        print(f"\n########## {ds} | NO CARGA: {e} ##########", flush=True)
        continue
    C, d = int(data["num_classes"]), int(data["input_dim"])
    tipo = "binario" if C == 2 else "multiclase"
    print(f"\n########## {ds} | {tipo} | C={C} d={d} ##########", flush=True)

    for emb in a.embedders:
        for ipc in a.ipcs:
            P = p_de(ipc)
            filas_min = C * ipc + C * 128    # lo que cruza el embedder por iteracion
            for s in range(a.seeds):
                seed = 400 + s
                # Las dos versiones seguidas sobre la misma semilla: lo que la
                # maquina este haciendo le toca igual a las dos.
                for ver, mod in VERSIONES:
                    if (ds, emb, ipc, ver, seed) in YA:
                        continue
                    cfg = dict(device=DEV, ipc=ipc, dm_iters=a.iters, dm_lr=0.5,
                               dm_batch_real=128, dm_embedder_type=emb,
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
                    except torch.cuda.OutOfMemoryError as e:
                        suelta()
                        torch.cuda.empty_cache()
                        print(f"  {ds:12s} {emb:11s} ipc{ipc:<4d} semilla {seed} "
                              f"{ver:6s}  SIN MEMORIA, se salta", flush=True)
                        continue
                    finally:
                        suelta()
                    reloj.marca()            # cierra la ultima iteracion
                    ms = reloj.ms()
                    dt = time.perf_counter() - t0
                    mem = torch.cuda.max_memory_allocated() / 2 ** 20

                    fila = dict(version=ver, dataset=ds, tipo=tipo, C=C, d=d,
                                embedder=emb, ipc=ipc, p=P,
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
                            w.writerow(["dataset", "tipo", "C", "embedder", "ipc",
                                        "version", "seed", "it", "ms"])
                        for i, v in enumerate(ms):
                            w.writerow([ds, tipo, C, emb, ipc, ver, seed, i,
                                        round(v, 4)])

                    n_hechas += 1
                    transcurrido = time.perf_counter() - t_inicio
                    falta = transcurrido / n_hechas * (len(pend) - n_hechas)
                    print(f"  {ds:12s} {emb:11s} ipc{ipc:<4d} semilla {seed} "
                          f"{ver:6s}  {dt:7.1f}s  "
                          f"{fila['ms_mediana']:7.2f} ms/iter  "
                          f"{mem:7.1f} MiB  |  {n_hechas}/{len(pend)}  "
                          f"faltan {falta/60:.0f} min", flush=True)

            # El par, en cuanto las dos versiones de esta celda estan.
            if len(VERSIONES) == 2:
                with open(CSV_RUNS) as f:
                    r = [x for x in csv.DictReader(f)
                         if x["dataset"] == ds and x["embedder"] == emb
                         and int(x["ipc"]) == ipc]
                med = lambda v, col: (sum(float(x[col]) for x in r
                                          if x["version"] == v)
                                      / max(sum(1 for x in r if x["version"] == v), 1))
                tv, tn = med("viejo", "total_s"), med("nuevo", "total_s")
                mv, mn = med("viejo", "pico_mib"), med("nuevo", "pico_mib")
                if tv and tn:
                    print(f"  {ds:12s} {emb:11s} ipc{ipc:<4d} -> tiempo "
                          f"{tv:7.1f}s vs {tn:7.1f}s = {tv/max(tn,1e-9):5.2f}x   "
                          f"memoria {mv:7.1f} vs {mn:7.1f} MiB = "
                          f"{mn/max(mv,1e-9):5.2f}x", flush=True)

print(f"\nCSV -> {CSV_RUNS}")
print(f"CSV -> {CSV_ITERS}")
