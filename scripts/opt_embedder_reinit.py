"""Building an embedder every iteration versus redrawing one in place.

TAME never trains the embedder: it samples one, uses it for a single iteration and
throws it away, so a full run builds `dm_iters * dm_views` networks. Each call to
`sample_random_embedder` constructs the module in Python, allocates every parameter
on the host, runs its initialiser there, and then copies each tensor to the device
one by one.

None of that is needed. The method wants new *weights*, not a new object.
`reinit_embedder_` redraws the parameters of an existing module in place, calling
the same `reset_parameters` PyTorch itself calls at construction, plus the explicit
initialisation of the raw `nn.Parameter` tensors of the tree ensemble.

Measured here per embedder family and size rung, repeated for an average:

  - wall time of building versus redrawing
  - how many parameter tensors and how many host-to-device copies that implies
  - that the redrawn weights come from the same distribution as the built ones

Writes ~/tame_runs/opt/embedder_reinit.csv and .../embedder_reinit_dist.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/zxxz6/TAME")
os.chdir("/home/zxxz6/TAME")
from models.embedders import (ObliviousTreeEnsemble, reinit_embedder_,
                              sample_random_embedder)

ap = argparse.ArgumentParser()
ap.add_argument("--embedders", nargs="+", default=["ln_res_l", "dcnv2_base", "node"])
ap.add_argument("--sizes", nargs="+", default=["base"])
# Los tres presupuestos del paper. La dimension del embedding sale del IPC con la
# misma tabla que usa el barrido, asi que cada linea corresponde a una celda real
# del experimento y no a un par de dimensiones inventado.
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--d", type=int, default=16,
                help="dimension de entrada; 16 es la de letter")
ap.add_argument("--reps", type=int, default=10,
                help="corridas por embedder y por modo; la tabla imprime la mediana")
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/opt"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PAPER_P = {10: 8, 25: 24, 50: 48, 100: 96, 150: 148, 200: 196}


def p_de(ipc):
    """La dimension del embedding que el barrido usa para ese presupuesto."""
    return max(4, min(PAPER_P.get(ipc, max(4, ipc - 4)), 256, ipc - 1))


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


def cronometra(fn, n):
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


filas, filas_dist = [], []
for emb in a.embedders:
    for size in a.sizes:
        for ipc in a.ipcs:
            d, p = a.d, p_de(ipc)
            torch.manual_seed(0)
            net = sample_random_embedder(emb, size, d, 256, p, DEV)
            n_t = sum(1 for _ in net.parameters())
            n_p = sum(q.numel() for q in net.parameters())

            construir = lambda: sample_random_embedder(emb, size, d, 256, p, DEV)
            reiniciar = lambda: reinit_embedder_(net)
            cronometra(construir, 3)
            cronometra(reiniciar, 3)

            for rep in range(a.reps):
                for modo, fn in [("construir", construir), ("reiniciar", reiniciar)]:
                    ms = cronometra(fn, 1)
                    filas.append(dict(embedder=emb, size=size, ipc=ipc, d=d, p=p,
                                      n_tensores=n_t, n_params=n_p,
                                      modo=modo, rep=rep, ms=round(ms, 4),
                                      **_estado_maquina()))
            g = pd.DataFrame(filas)
            g = g[(g.embedder == emb) & (g.d == d) & (g.p == p)]
            gc, gr = g[g.modo == "construir"].ms, g[g.modo == "reiniciar"].ms
            c, r = gc.median(), gr.median()
            print(f"{emb:11s} ipc={ipc:<3d} p={p:3d}  {n_t:2d} tensores "
                  f"{n_p:10,d} params  | {len(gc):2d} corridas  "
                  f"construir {c:7.2f} +-{gc.std():.2f} ms   "
                  f"reiniciar {r:5.2f} +-{gr.std():.2f} ms   {c/max(r,1e-9):6.1f}x   "
                  f"ahorro {c-r:6.2f} s por 1000 iters", flush=True)

            # La distribucion de los pesos tiene que coincidir, o el reciclado
            # estaria cambiando el metodo y no solo su costo.
            x = torch.randn(256, d, device=DEV)
            for modo in ("construir", "reiniciar"):
                for k in range(10):
                    if modo == "construir":
                        m = sample_random_embedder(emb, size, d, 256, p, DEV)
                    else:
                        reinit_embedder_(net)
                        m = net
                    with torch.no_grad():
                        z = m(x)
                    pesos = torch.cat([q.flatten() for q in m.parameters()])
                    filas_dist.append(dict(
                        embedder=emb, ipc=ipc, d=d, p=p, modo=modo, rep=k,
                        peso_std=float(pesos.std()), peso_media=float(pesos.mean()),
                        salida_std=float(z.std()), salida_media=float(z.mean()),
                        salida_min=float(z.min()), salida_max=float(z.max())))

pd.DataFrame(filas).to_csv(f"{a.out}/embedder_reinit.csv", index=False)
pd.DataFrame(filas_dist).to_csv(f"{a.out}/embedder_reinit_dist.csv", index=False)
print(f"\nCSV -> {a.out}/embedder_reinit.csv")
print(f"CSV -> {a.out}/embedder_reinit_dist.csv")
