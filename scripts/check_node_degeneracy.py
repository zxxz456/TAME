"""Rank check for the NODE embedder, with and without random tree init.

TAME never trains the embedder, so the zero-initialised feature_logits and
thresholds stay zero: softmax(zeros) is uniform, every tree and level reads the
same mean of all coordinates, and the ensemble collapses. This measures that
directly, and confirms whether tree_random_init undoes it.

Effective rank near d+1 means the trees contribute one dimension on top of the
linear residual path, i.e. the fix did not take.
"""
import sys, os
import numpy as np, torch
sys.path.insert(0, "/home/zxxz6/TAME"); os.chdir("/home/zxxz6/TAME")
from models.embedders import build_embedder

D, P, N = 20, 128, 512          # input dim, embed dim, samples
TREES, DEPTH, TDIM, LAYERS = 64, 6, 24, 2

def eff_rank(Z, tol=1e-3):
    sv = torch.linalg.svdvals(Z - Z.mean(0, keepdim=True))
    return int((sv > tol * sv[0]).sum()), sv

torch.manual_seed(0)
x = torch.randn(N, D)

for tag, rnd in [("zero init (publicado)", False), ("random init (fix)", True)]:
    torch.manual_seed(0)
    net = build_embedder("node", input_dim=D, hidden=256, embed_dim=P,
                         num_layers=LAYERS, num_trees=TREES, depth=DEPTH,
                         tree_dim=TDIM, dropout=0.0, tree_random_init=rnd)
    for q in net.parameters():
        q.requires_grad_(False)
    net.eval()

    ens = net.layers[0]
    h = net.ln(net.in_proj(x))
    sel = torch.softmax(ens.feature_logits, dim=-1)      # (T, D, tdim)
    x_sel = torch.einsum("bi,tdi->btd", h, sel)          # (B, T, D)

    with torch.no_grad():
        z = net(x)                 # embedder completo
        e = ens(h)                 # SOLO el ensamble de arboles
    r_all, _ = eff_rank(z)
    r_ens, sv_ens = eff_rank(e)

    print(f"\n--- {tag} ---")
    print(f"  std de x_sel entre arboles y niveles   : {x_sel.std(dim=(1,2)).mean():.6f}")
    print(f"  |x_sel[0,0,0] - mean(h[0])|            : {abs(x_sel[0,0,0] - h[0].mean()):.3e}")
    print(f"  rank del ENSAMBLE solo                 : {r_ens} de {TDIM}   <-- lo que mide el fix")
    print(f"  5 val. singulares del ensamble         : {[round(v,3) for v in sv_ens[:5].tolist()]}")
    print(f"  rank del embedder completo             : {r_all} de {P}"
          f"   (techo real = tree_dim = {TDIM}, con in_proj de rango {D})")
