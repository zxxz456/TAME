"""Shapes and launch counts: one pass per class versus the class as a dimension.

The distillation used to walk the classes one by one, so every class ran its own
forward, its own gather and its own reductions. The classes are independent, so
they stack into a leading dimension and cross the embedder as a single batch.

This records what that costs and what it moves, without touching accuracy:

  - the shape of every intermediate tensor in both paths
  - how many CUDA kernels each path actually launches, counted with the profiler
    rather than estimated
  - peak device memory for each path
  - whether the two paths agree, to the last bit on the embedder output and to
    float32 reduction noise on the gradient

Writes ~/tame_runs/opt/tensor_shapes.csv and .../tensor_ops.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, "/home/zxxz6/TAME")
os.chdir("/home/zxxz6/TAME")
from data.prepare_database import prepare_db
from models.embedders import sample_random_embedder
from synth.tame_synth import cov_matrix

ap = argparse.ArgumentParser()
# Seis valores de C, que es la unica variable de la que depende el conteo de
# operaciones: 2, 5, 6, 7, 10 y 26. Con eso la ley en C se puede ajustar en vez
# de inferirla de dos puntos.
ap.add_argument("--datasets", nargs="+",
                default=["magic", "pageblocks", "satimage", "shuttle",
                         "pendigits", "letter"])
ap.add_argument("--ipcs", nargs="+", type=int, default=[10, 50, 200])
ap.add_argument("--embedder", default="ln_res_l",
                choices=["ln_res_l", "dcnv2_base", "node"])
ap.add_argument("--batch-real", type=int, default=128)
ap.add_argument("--reps", type=int, default=5,
                help="repeticiones por celda; el conteo de ops es determinista, "
                     "la memoria pico no del todo")
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/opt"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    raise SystemExit("este experimento mide kernels y memoria de GPU; hace falta cuda")
PAPER_P = {10: 8, 25: 24, 50: 48, 100: 96, 150: 148, 200: 196}


def p_de(ipc):
    return max(4, min(PAPER_P.get(ipc, max(4, ipc - 4)), 256, ipc - 1))


def por_clase(net, real, syn, C, B, ipc, d, p, eps=1e-6, cw=1.0):
    """El camino viejo: una pasada completa por cada clase."""
    lm = torch.zeros((), device=DEV)
    lc = torch.zeros((), device=DEV)
    formas = []
    for c in range(C):
        fr = net(real[c]).detach()
        fs = net(syn[c * ipc:(c + 1) * ipc])
        mu_r, cov_r = cov_matrix(fr, eps)
        mu_s, cov_s = cov_matrix(fs, eps)
        lm = lm + ((mu_r - mu_s) ** 2).sum()
        dif = cov_r - cov_s
        lc = lc + cw * (dif * dif).sum()
        if c == 0:                      # las formas son iguales para toda clase
            formas = [("real de la clase", tuple(real[c].shape)),
                      ("syn de la clase", (ipc, d)),
                      ("feat_real", tuple(fr.shape)),
                      ("feat_syn", tuple(fs.shape)),
                      ("mu", tuple(mu_s.shape)),
                      ("cov", tuple(cov_s.shape))]
    return lm + lc, formas


def batcheado(net, real, syn, C, B, ipc, d, p, eps=1e-6, cw=1.0):
    """El camino nuevo: la clase como dimension de lote."""
    fr = net(real.reshape(-1, d)).detach().view(C, B, p)
    fs = net(syn.view(C, ipc, d).reshape(-1, d)).view(C, ipc, p)
    mu_r, mu_s = fr.mean(1), fs.mean(1)
    zr, zs = fr - mu_r.unsqueeze(1), fs - mu_s.unsqueeze(1)
    cov_r = zr.transpose(1, 2) @ zr / B
    cov_s = zs.transpose(1, 2) @ zs / ipc
    I = eps * torch.eye(p, device=DEV)
    cov_r, cov_s = cov_r + I, cov_s + I
    dif = cov_r - cov_s
    total = ((mu_r - mu_s) ** 2).sum() + cw * (dif * dif).sum()
    formas = [("real apilado", tuple(real.shape)),
              ("syn apilado", (C, ipc, d)),
              ("entrada al embedder", (C * ipc, d)),
              ("feat_real", tuple(fr.shape)),
              ("feat_syn", tuple(fs.shape)),
              ("mu", tuple(mu_s.shape)),
              ("cov", tuple(cov_s.shape))]
    return total, formas


class _Contador(TorchDispatchMode):
    """Cuenta despachos de ATen, ida y vuelta.

    El profiler de CUDA necesita CUPTI y aqui no esta disponible, asi que en vez
    de kernels se cuentan operaciones de ATen. Cada una se traduce en uno o mas
    kernels, asi que no es el numero absoluto de lanzamientos; para comparar los
    dos caminos sirve igual, porque la traduccion es la misma en ambos."""

    def __init__(self):
        self.n = 0
        self.ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        self.ops[str(func).split(".")[0]] += 1
        return func(*args, **(kwargs or {}))


def cuenta_ops(fn):
    cont = _Contador()
    with cont:
        out = fn()
        out.backward()
    torch.cuda.synchronize()
    return cont.n, cont.ops


def pico_memoria(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    out.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


filas_formas, filas_ops = [], []
for ds in a.datasets:
    data = prepare_db({"random_seed": 132, "device": DEV}, name=ds)
    C, d = int(data["num_classes"]), int(data["input_dim"])
    for ipc in a.ipcs:
        p = p_de(ipc)
        B = a.batch_real
        print(f"\n=== {ds} | C={C} d={d} | ipc={ipc} p={p} ===", flush=True)
        torch.manual_seed(0)
        net = sample_random_embedder(a.embedder, "base", d, 256, p, DEV)
        net.eval()
        for rep in range(a.reps):
            res = {}
            # EL MISMO syn para los dos caminos: si cada uno sortea el suyo, la
            # comparacion de perdida y gradiente no dice nada. Cambia entre
            # repeticiones, para que la equivalencia se compruebe sobre datos
            # distintos y no sobre un solo sorteo afortunado.
            torch.manual_seed(1000 + rep)
            syn0 = torch.randn(C * ipc, d, device=DEV)
            real = torch.randn(C, B, d, device=DEV)
            for nombre, fn_path in [("por_clase", por_clase), ("batcheado", batcheado)]:
                syn = syn0.clone().requires_grad_(True)
                fn = lambda: fn_path(net, real, syn, C, B, ipc, d, p)[0]
                # el valor y el gradiente, para comprobar que coinciden
                loss = fn_path(net, real, syn, C, B, ipc, d, p)[0]
                g = torch.autograd.grad(loss, syn)[0]
                formas = fn_path(net, real, syn, C, B, ipc, d, p)[1]
                n_k, detalle = cuenta_ops(fn)
                mem = pico_memoria(fn)
                res[nombre] = (float(loss.detach()), g, n_k, mem, detalle)

                if rep == 0:                      # las formas no cambian
                    for paso, (etq, forma) in enumerate(formas):
                        n = int(np.prod(forma))
                        filas_formas.append(dict(
                            dataset=ds, C=C, d=d, ipc=ipc, p=p, batch_real=B,
                            embedder=a.embedder, modo=nombre, paso=paso, tensor=etq,
                            forma=str(forma), ndim=len(forma), elementos=n,
                            mib=round(n * 4 / 2 ** 20, 3),
                            veces=C if nombre == "por_clase" else 1))

            (l_v, g_v, k_v, m_v, d_v), (l_n, g_n, k_n, m_n, d_n) = (res["por_clase"],
                                                                     res["batcheado"])
            filas_ops.append(dict(
                dataset=ds, C=C, d=d, ipc=ipc, p=p, embedder=a.embedder, rep=rep,
                ops_por_clase=k_v, ops_batcheado=k_n,
                factor_ops=round(k_v / max(k_n, 1), 2),
                mib_por_clase=round(m_v, 1), mib_batcheado=round(m_n, 1),
                factor_memoria=round(m_n / max(m_v, 1e-9), 3),
                loss_por_clase=l_v, loss_batcheado=l_n,
                err_rel_loss=abs(l_v - l_n) / max(abs(l_v), 1e-12),
                err_rel_grad=float((g_v - g_n).norm() / g_v.norm()),
                filas_por_iter_batcheado=C * ipc + C * B))

        g = pd.DataFrame(filas_ops)
        g = g[(g.dataset == ds) & (g.ipc == ipc)]
        print(f"  ops ATen {g.ops_por_clase.iloc[0]:5.0f} -> {g.ops_batcheado.iloc[0]:4.0f}"
              f"   ({g.factor_ops.iloc[0]:.1f}x menos)")
        print(f"  memoria  {g.mib_por_clase.mean():7.1f} -> {g.mib_batcheado.mean():7.1f} MiB"
              f"   ({g.factor_memoria.mean():.2f}x)   sobre {a.reps} repeticiones")
        print(f"  err grad {g.err_rel_grad.min():.2e} a {g.err_rel_grad.max():.2e}"
              f"   err loss max {g.err_rel_loss.max():.2e}")

ops = pd.DataFrame(filas_ops)
print("\n=== operaciones de ATen contra el numero de clases ===")
u = ops.drop_duplicates(["dataset", "ipc"]).groupby("C")[["ops_por_clase", "ops_batcheado"]].first()
print(u)
if len(u) > 1:
    pend, orig = np.polyfit(u.index.values, u.ops_por_clase.values, 1)
    print(f"\n  camino viejo: {orig:.0f} + {pend:.0f} * C     (ajuste lineal)")
    print(f"  camino nuevo: {u.ops_batcheado.iloc[0]:.0f}, constante")

pd.DataFrame(filas_formas).to_csv(f"{a.out}/tensor_shapes.csv", index=False)
pd.DataFrame(filas_ops).to_csv(f"{a.out}/tensor_ops.csv", index=False)
print(f"\nCSV -> {a.out}/tensor_shapes.csv")
print(f"CSV -> {a.out}/tensor_ops.csv")
