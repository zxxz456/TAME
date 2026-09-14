"""Log the DM term and the adversarial term separately, to see whether the
contamination is large enough to matter at all."""
import os, sys, random
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/zxxz6/TAME"); os.chdir("/home/zxxz6/TAME")
from data.prepare_database import prepare_db
from models.embedders import sample_random_embedder
from synth.tame_synth import cov_matrix
from synth.tame_synth_critic import CriticMLP, _gradient_penalty

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IPC, ITERS, ADV_W, N_CRITIC = 50, 200, 0.05, 3
random.seed(132); np.random.seed(132); torch.manual_seed(132)

data = prepare_db({"random_seed": 132, "device": DEV}, name="adult")
X, y = data["X_train"].to(DEV).float(), data["y_train"].to(DEV).long()
d, C = data["input_dim"], data["num_classes"]
idx_c = [np.where(y.cpu().numpy() == c)[0] for c in range(C)]
rb = lambda c, n: X[np.random.choice(idx_c[c], n, replace=len(idx_c[c]) < n)]

syn = torch.randn((C * IPC, d), device=DEV, requires_grad=True)
lab = torch.arange(C, device=DEV).repeat_interleave(IPC)
with torch.no_grad():
    for c in range(C):
        syn[c*IPC:(c+1)*IPC] = rb(c, IPC)
opt = torch.optim.SGD([syn], lr=0.5, momentum=0.5)
critic = CriticMLP(d, C).to(DEV)
optc = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

rows = []
for it in range(ITERS + 1):
    net = sample_random_embedder("ln_res_l", "base", d, 256, 48, DEV); net.eval()
    critic.train()
    for _ in range(N_CRITIC):
        optc.zero_grad(set_to_none=True)
        yb = torch.randint(0, C, (128,), device=DEV)
        xr = torch.cat([rb(int(c), 1) for c in yb.cpu().numpy()])
        ids = torch.tensor([int(c)*IPC + np.random.randint(0, IPC) for c in yb.tolist()],
                           device=DEV, dtype=torch.long)
        xs = syn[ids].detach()
        ld = critic(xs, yb).mean() - critic(xr, yb).mean()
        (ld + _gradient_penalty(critic, xr, yb, xs, 10.0)).backward()
        optc.step()

    opt.zero_grad(set_to_none=True)
    lm = torch.zeros((), device=DEV); lc = torch.zeros((), device=DEV)
    for c in range(C):
        fr = net(rb(c, 128)).detach(); fs = net(syn[c*IPC:(c+1)*IPC])
        mr, cr = cov_matrix(fr, 1e-6); ms, cs = cov_matrix(fs, 1e-6)
        lm = lm + ((mr - ms) ** 2).sum(); lc = lc + ((cr - cs) ** 2).sum()
    loss_dm = lm + lc
    critic.eval()
    d_syn = critic(syn, lab).mean()
    adv = ADV_W * (-d_syn)

    # Gradient norms of each term separately. The value ratio answers whether the
    # adversarial term can move the argmin of the selection scalar; this answers
    # the different and more important question of whether it moves the data at
    # all. A small-valued term can still be steep.
    g_dm = torch.autograd.grad(loss_dm, syn, retain_graph=True)[0].norm().item()
    g_adv = torch.autograd.grad(adv, syn, retain_graph=True)[0].norm().item()

    (loss_dm + adv).backward()
    torch.nn.utils.clip_grad_norm_([syn], 10.0); opt.step()

    rows.append(dict(it=it, loss_dm=loss_dm.item(), adv=adv.item(),
                     d_syn=d_syn.item(),
                     ratio=abs(adv.item()) / max(abs(loss_dm.item()), 1e-12),
                     g_dm=g_dm, g_adv=g_adv,
                     g_ratio=g_adv / max(g_dm, 1e-12)))
    if it % 100 == 0:
        r = rows[-1]
        print(f"it {it:4d} | dm {r['loss_dm']:9.4f} | adv {r['adv']:9.4f} "
              f"| D(syn) {r['d_syn']:8.4f} | |adv|/dm {r['ratio']:6.3f} "
              f"| grad dm {r['g_dm']:8.4f} adv {r['g_adv']:8.4f} "
              f"| ratio {r['g_ratio']:6.3f}", flush=True)

df = pd.DataFrame(rows)
out = os.path.expanduser("~/tame_runs/critic_terms.csv"); df.to_csv(out, index=False)
print("\n=== magnitudes (valor) ===")
print(df[["loss_dm", "adv", "d_syn", "ratio"]].describe().loc[["mean","50%","min","max"]].round(4))
print("\n=== magnitudes (gradiente sobre syn_data) ===")
print(df[["g_dm", "g_adv", "g_ratio"]].describe().loc[["mean","50%","min","max"]].round(4))
print("\n=== deriva de D(syn) por tramo ===")
print(df.groupby(df.it // 200).d_syn.agg(["mean","min","max"]).round(4))
print("\nCSV ->", out)
