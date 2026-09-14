"""The optimised core against the published one: same distributions, less time.

The three changes (redrawing the embedder in place, stacking the classes into a
leading dimension, and keeping the running minimum on the device) are meant to cost
less without changing what the method computes. "Without changing" cannot mean
bit-identical: the published code seeds every embedder from the wall clock, so two
of its own runs with the same seed already differ. The claim that can be tested is
that the *distribution* of results is the same.

The previous version is read straight out of the commit that holds it, so nothing
here depends on the working tree still containing it.

For each embedder, dataset and budget, both versions are run over several seeds and
two things are recorded:

  tiempo   total seconds, milliseconds per iteration and its spread, so the saving
           is separated from the noise
  forma    the distribution of the distilled set and of its projection through a
           fresh embedder: moments, quantiles, and how far each row moved from the
           real rows it started at

Writes ~/tame_runs/opt/equiv_runs.csv, .../equiv_quantiles.csv
and .../equiv_raw.npz with the distilled sets themselves.
"""
import argparse
import importlib.util
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, "/home/zxxz6/TAME")
os.chdir("/home/zxxz6/TAME")
from data.prepare_database import prepare_db
from models.embedders import sample_random_embedder

ap = argparse.ArgumentParser()
ap.add_argument("--binario", default="magic")
ap.add_argument("--multiclase", default="pageblocks")
ap.add_argument("--embedders", nargs="+", default=["ln_res_l", "dcnv2_base", "node"])
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--commit", default="ac15fc1",
                help="commit con el nucleo previo a la optimizacion")
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/opt"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    raise SystemExit("hace falta cuda: medir tiempos en cpu no dice nada del cambio")
PAPER_P = {10: 8, 25: 24, 50: 48, 100: 96, 150: 148, 200: 196}
QS = np.round(np.linspace(0, 1, 21), 3)

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
print(f"[versiones] viejo desde {a.commit}  |  nuevo desde el arbol de trabajo")
print(f"[dispositivo] {DEV} ({torch.cuda.get_device_name(0)})")


def _estado_maquina():
    """Condiciones en las que se tomo la medida.

    El tiempo de pared depende de que mas este corriendo, y la contencion no
    afecta por igual a los dos brazos: construir un embedder reserva memoria de
    GPU y transfiere, y eso se serializa entre procesos, mientras que reiniciarlo
    no reserva nada. Sin este registro, un cociente medido con la maquina llena no
    se distingue de uno medido con ella vacia."""
    import subprocess
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


def p_de(ipc):
    return max(4, min(PAPER_P.get(ipc, max(4, ipc - 4)), 256, ipc - 1))


def resumen(X, y, data, emb, p, seed):
    """Momentos y cuantiles del conjunto destilado y de su proyeccion.

    El embedder de medicion se sortea con una semilla fija e independiente de la
    destilacion, para que las dos versiones se midan con la misma regla."""
    x = X.detach().float()
    torch.manual_seed(9_000 + seed)          # la misma regla para ambas versiones
    net = sample_random_embedder(emb, "base", int(data["input_dim"]), 256, p, DEV)
    net.eval()
    with torch.no_grad():
        z = net(x)
    xf, zf = x.flatten().cpu().numpy(), z.flatten().cpu().numpy()
    out = dict(
        x_media=float(x.mean()), x_std=float(x.std()),
        x_min=float(x.min()), x_max=float(x.max()),
        x_skew=float(stats.skew(xf)), x_kurt=float(stats.kurtosis(xf)),
        z_media=float(z.mean()), z_std=float(z.std()),
        z_skew=float(stats.skew(zf)), z_kurt=float(stats.kurtosis(zf)),
        # Rango efectivo del conjunto destilado: cuantas direcciones ocupa de verdad
        rango=int(torch.linalg.matrix_rank(x - x.mean(0)).item()),
    )
    cuant = [dict(cual="x", q=float(q), valor=float(np.quantile(xf, q))) for q in QS]
    cuant += [dict(cual="z", q=float(q), valor=float(np.quantile(zf, q))) for q in QS]
    return out, cuant


filas, filas_q, crudos = [], [], {}
for ds in (a.binario, a.multiclase):
    data = prepare_db({"random_seed": 132, "device": DEV}, name=ds)
    tipo = "binario" if int(data["num_classes"]) == 2 else "multiclase"
    for emb in a.embedders:
        for ipc in a.ipcs:
            p = p_de(ipc)
            for ver, mod in [("viejo", VIEJO), ("nuevo", NUEVO)]:
                for s in range(a.seeds):
                    seed = 400 + s
                    cfg = dict(device=DEV, ipc=ipc, dm_iters=a.iters, dm_lr=0.5,
                               dm_batch_real=128, dm_embedder_type=emb,
                               dm_embedder_size="base", dm_embed_hidden=256,
                               dm_embed_dim=p, random_seed=seed)
                    set_seed(seed)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    X, y = mod.tame_synthesize(data, cfg)[:2]
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0

                    r, cu = resumen(X, y, data, emb, p, s)
                    clave = f"{ver}|{ds}|{emb}|{ipc}|{seed}"
                    crudos[clave] = X.detach().cpu().numpy()
                    filas.append(dict(version=ver, dataset=ds, tipo=tipo, C=int(data["num_classes"]),
                                      d=int(data["input_dim"]), embedder=emb, ipc=ipc, p=p,
                                      seed=seed, iters=a.iters,
                                      total_s=round(dt, 3),
                                      ms_por_iter=round(dt / (a.iters + 1) * 1000, 4),
                                      filas_syn=int(X.shape[0]), **r,
                                      **_estado_maquina()))
                    for c in cu:
                        filas_q.append(dict(version=ver, dataset=ds, embedder=emb,
                                            ipc=ipc, seed=seed, **c))
                g = pd.DataFrame(filas)
                g = g[(g.dataset == ds) & (g.embedder == emb) & (g.ipc == ipc)]
                v = g[g.version == "viejo"]
                n = g[g.version == "nuevo"]
                if len(v) and len(n):
                    print(f"{ds:11s} {emb:11s} ipc{ipc:<3d} p={p:3d}  "
                          f"viejo {v.total_s.mean():7.1f}s   nuevo {n.total_s.mean():7.1f}s   "
                          f"{v.total_s.mean()/max(n.total_s.mean(),1e-9):5.2f}x   "
                          f"std(X) {v.x_std.mean():.4f} vs {n.x_std.mean():.4f}",
                          flush=True)

df = pd.DataFrame(filas)
df.to_csv(f"{a.out}/equiv_runs.csv", index=False)
pd.DataFrame(filas_q).to_csv(f"{a.out}/equiv_quantiles.csv", index=False)
np.savez_compressed(f"{a.out}/equiv_raw.npz", **crudos)

print("\n=== tiempo ===")
t = df.pivot_table(index=["dataset", "embedder", "ipc"], columns="version",
                   values=["total_s", "ms_por_iter"], aggfunc="mean")
print(t.round(3))

print("\n=== forma de la distribucion, viejo contra nuevo ===")
for col in ["x_std", "x_skew", "x_kurt", "z_std", "z_kurt", "rango"]:
    w = df.pivot_table(index=["dataset", "embedder", "ipc", "seed"],
                       columns="version", values=col)
    dif = (w["nuevo"] - w["viejo"])
    p_val = stats.wilcoxon(w["nuevo"], w["viejo"]).pvalue if len(w) > 5 else float("nan")
    print(f"  {col:8s} viejo {w['viejo'].mean():9.4f}  nuevo {w['nuevo'].mean():9.4f}  "
          f"dif {dif.mean():+9.4f}  wilcoxon p={p_val:.3f}")

print(f"\nCSV -> {a.out}/equiv_runs.csv")
print(f"CSV -> {a.out}/equiv_quantiles.csv")
print(f"NPZ -> {a.out}/equiv_raw.npz")
