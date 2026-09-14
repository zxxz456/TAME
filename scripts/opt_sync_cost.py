"""What the per-iteration synchronisation costs.

A GPU does not execute when Python asks: it takes the kernel into a queue and the
host runs ahead enqueuing the next one. Reading a value back breaks that overlap,
because the value only exists once the GPU has drained everything queued. The host
stalls, and when the number finally arrives the queue is empty, so now the GPU
stalls while the host refills it.

The loop used to read one value per iteration to track the running minimum, and a
second one in the `if torch.isfinite(loss)` around the backward, since casting a
tensor to bool waits too. The critic variant added six more per iteration building
its real batch row by row.

Two settings are measured, both over the real shape of the distillation loop:

  aislado    only the moment arithmetic, so the stall is not hidden by other work
  embedder   the same with a real embedder forward and backward in the middle,
             which is the honest figure: with more work per iteration the stall
             weighs relatively less

Writes ~/tame_runs/opt/sync_cost.csv
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
from models.embedders import sample_random_embedder

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", default=["2:10:8", "2:50:48", "5:50:48",
                                                 "10:50:48", "26:50:48", "26:200:196"],
                help="C:ipc:p, el numero de clases, el presupuesto y la dim del embedding")
ap.add_argument("--d", type=int, default=16, help="dimension de entrada")
ap.add_argument("--batch-real", type=int, default=128)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--embedder", default="ln_res_l",
                choices=["ln_res_l", "dcnv2_base", "node"])
ap.add_argument("--out", default=os.path.expanduser("~/tame_runs/opt"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    raise SystemExit("sin cuda no hay cola que vaciar, la medicion no significa nada")


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


def loop(C, ipc, p, d, con_sync, con_embedder, net, iters):
    """Un loop con la forma del de TAME, con y sin lectura por iteracion."""
    syn = torch.randn(C * ipc, d, device=DEV, requires_grad=True)
    real = torch.randn(C, a.batch_real, d, device=DEV)
    opt = torch.optim.SGD([syn], lr=0.5, momentum=0.5)
    best_t = torch.full((), float("inf"), device=DEV)
    best_py = float("inf")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        if con_embedder:
            fr = net(real.reshape(-1, d)).detach().view(C, a.batch_real, p)
            fs = net(syn.view(C, ipc, d).reshape(-1, d)).view(C, ipc, p)
        else:
            fr = real[:, :, :1].expand(C, a.batch_real, p).contiguous()
            fs = syn.view(C, ipc, d)[:, :, :1].expand(C, ipc, p)
        mu_r, mu_s = fr.mean(1), fs.mean(1)
        zr, zs = fr - mu_r.unsqueeze(1), fs - mu_s.unsqueeze(1)
        cov_r = zr.transpose(1, 2) @ zr / a.batch_real
        cov_s = zs.transpose(1, 2) @ zs / ipc
        dif = cov_r - cov_s
        loss = ((mu_r - mu_s) ** 2).sum() + (dif * dif).sum()

        if con_sync:
            # el patron viejo: la CPU se para aqui, dos veces por iteracion
            cur = float((loss / C).detach().item())
            if np.isfinite(cur) and cur < best_py:
                best_py = cur
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_([syn], 10.0)
                opt.step()
        else:
            # el patron nuevo: la decision se queda en el dispositivo
            cur = (loss / C).detach()
            best_t = torch.where(torch.isfinite(cur) & (cur < best_t), cur, best_t)
            loss.backward()
            torch.nan_to_num_(syn.grad, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nn.utils.clip_grad_norm_([syn], 10.0)
            opt.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


filas = []
for cfg in a.configs:
    C, ipc, p = (int(x) for x in cfg.split(":"))
    torch.manual_seed(0)
    net = sample_random_embedder(a.embedder, "base", a.d, 256, p, DEV)
    net.eval()
    for escenario, con_emb in [("aislado", False), ("embedder", True)]:
        for con_sync in (True, False):
            loop(C, ipc, p, a.d, con_sync, con_emb, net, 20)      # calentar
            for rep in range(a.reps):
                ms = loop(C, ipc, p, a.d, con_sync, con_emb, net, a.iters)
                filas.append(dict(C=C, ipc=ipc, p=p, d=a.d, batch_real=a.batch_real,
                                  embedder=a.embedder, escenario=escenario,
                                  modo="con_sync" if con_sync else "sin_sync",
                                  rep=rep, iters=a.iters, ms_por_iter=round(ms, 4),
                                  **_estado_maquina()))
        g = pd.DataFrame(filas)
        g = g[(g.C == C) & (g.ipc == ipc) & (g.escenario == escenario)]
        cs = g[g.modo == "con_sync"].ms_por_iter.mean()
        ss = g[g.modo == "sin_sync"].ms_por_iter.mean()
        print(f"C={C:2d} ipc={ipc:3d} p={p:3d}  {escenario:9s}  "
              f"con sync {cs:8.3f} ms   sin sync {ss:8.3f} ms   "
              f"{cs/ss:5.2f}x   ahorro {(cs-ss)*1000:7.0f} us/iter", flush=True)

df = pd.DataFrame(filas)
df.to_csv(f"{a.out}/sync_cost.csv", index=False)
print("\n=== resumen (media de las repeticiones) ===")
r = df.pivot_table(index=["escenario", "C", "ipc", "p"], columns="modo",
                   values="ms_por_iter", aggfunc="mean")
r["factor"] = (r.con_sync / r.sin_sync).round(2)
r["ahorro_s_por_1000_iters"] = ((r.con_sync - r.sin_sync)).round(2)
print(r.round(3))
print(f"\nCSV -> {a.out}/sync_cost.csv")
