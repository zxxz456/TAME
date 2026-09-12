"""
TAME + adversarial critic: mean+cov2 distribution matching with optional
center loss and WGAN-GP critic regularizer.

Register as "tame_critic" in synth/registry.py.

Config keys (beyond base TAME):
  dm_use_center, dm_center_weight
  dm_use_critic, dm_adv_weight, dm_n_critic, dm_critic_lr, dm_gp_lambda
"""

import os
import torch
import torch.nn as nn
import numpy as np
from models.embedders import sample_random_embedder
from .tame_synth import cov_matrix


class CriticMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256, depth=3, dropout=0.0):
        super().__init__()
        y_emb_dim = min(32, hidden)
        self.y_emb = nn.Embedding(num_classes, y_emb_dim)
        layers = []
        d = input_dim + y_emb_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.2, inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, self.y_emb(y)], dim=1)).squeeze(1)


def _gradient_penalty(D, x_real, y, x_fake, lam=10.0):
    b = x_real.size(0)
    eps = torch.rand(b, 1, device=x_real.device)
    x_hat = (eps * x_real + (1 - eps) * x_fake).requires_grad_(True)
    d_hat = D(x_hat, y)
    grads = torch.autograd.grad(d_hat.sum(), x_hat, create_graph=True, retain_graph=True)[0]
    return lam * ((grads.norm(2, dim=1) - 1) ** 2).mean()


def tame_critic_synthesize(data, config):
    device = config["device"]
    X_train = data["X_train"].to(device).float()
    y_train = data["y_train"].to(device).long()

    ipc = int(config["ipc"])
    iters = int(config["dm_iters"])
    lr = float(config["dm_lr"])
    batch_real = int(config["dm_batch_real"])
    input_dim = int(data["input_dim"])
    num_classes = int(data["num_classes"])

    embed_hidden = int(config["dm_embed_hidden"])
    embed_dim = int(config["dm_embed_dim"])
    embedder_type = config["dm_embedder_type"]
    embedder_size = config.get("dm_embedder_size", "base")

    grad_clip = float(config.get("grad_clip", 10.0))
    eps = float(config.get("moment_eps", 1e-6))
    cov_weight = float(config.get("cov_weight", 1.0))
    save_dir = config.get("save_dir", None)

    # center loss
    use_center = bool(config.get("dm_use_center", False))
    center_weight = float(config.get("dm_center_weight", 0.1))

    # critic
    use_critic = bool(config.get("dm_use_critic", False))
    adv_weight = float(config.get("dm_adv_weight", 0.05))
    n_critic = int(config.get("dm_n_critic", 3))
    critic_lr = float(config.get("dm_critic_lr", 1e-4))
    gp_lambda = float(config.get("dm_gp_lambda", 10.0))
    critic_warmup = int(config.get("dm_critic_warmup", 0))

    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        idx = indices_class[c]
        return X_train[np.random.choice(idx, n, replace=len(idx) < n)]

    syn_data = torch.randn((num_classes * ipc, input_dim), device=device, requires_grad=True)
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc)

    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c * ipc:(c + 1) * ipc] = get_real_batch(c, ipc)

    prototypes = None
    if use_center:
        prototypes = torch.zeros((num_classes, embed_dim), device=device, requires_grad=True)

    params = [syn_data] + ([prototypes] if use_center else [])
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.5)

    critic = None
    opt_critic = None
    if use_critic:
        critic = CriticMLP(input_dim, num_classes).to(device)
        opt_critic = torch.optim.Adam(critic.parameters(), lr=critic_lr, betas=(0.5, 0.9))

    # Optional snapshot trail, mirroring tame_synth.py. This is instrumentation,
    # not intervention: cloning consumes no randomness and touches neither the
    # optimizer nor best_syn, so the distillation trajectory is unchanged.
    # Recording the trail is what lets the critic arm take part in the
    # validation-based selection of Sec. 4.2, which it could not before.
    snapshot_every = int(config.get("snapshot_every", 0))
    snapshots = []
    if snapshot_every:
        snapshots.append((-1, syn_data.detach().clone()))   # the init subset

    best_loss = float("inf")
    best_it = -1
    best_syn = syn_data.detach().clone()

    for it in range(iters + 1):
        embed_net = sample_random_embedder(
            embedder_type, embedder_size, input_dim, embed_hidden, embed_dim, device
        )
        embed_net.eval()

        # critic update
        critic_active = use_critic and it >= critic_warmup
        if critic_active:
            critic.train()
            for _ in range(n_critic):
                opt_critic.zero_grad(set_to_none=True)
                yb = torch.randint(0, num_classes, (batch_real,), device=device)
                xb_real = torch.cat([get_real_batch(int(c), 1) for c in yb.cpu().numpy()])
                idxs = torch.tensor([int(c) * ipc + np.random.randint(0, ipc) for c in yb.tolist()],
                                    device=device, dtype=torch.long)
                xb_syn = syn_data[idxs].detach()
                loss_D = critic(xb_syn, yb).mean() - critic(xb_real, yb).mean()
                loss_D = loss_D + _gradient_penalty(critic, xb_real, yb, xb_syn, gp_lambda)
                loss_D.backward()
                opt_critic.step()

        # syn update
        optimizer.zero_grad(set_to_none=True)

        loss_mean = torch.zeros((), device=device)
        loss_cov = torch.zeros((), device=device)
        loss_center = torch.zeros((), device=device)

        for c in range(num_classes):
            real_b = get_real_batch(c, batch_real)
            syn_b = syn_data[c * ipc:(c + 1) * ipc]

            feat_real = embed_net(real_b).detach()
            feat_syn = embed_net(syn_b)

            mu_r, cov_r = cov_matrix(feat_real, eps)
            mu_s, cov_s = cov_matrix(feat_syn, eps)

            loss_mean = loss_mean + ((mu_r - mu_s) ** 2).sum()
            diff = cov_r - cov_s
            loss_cov = loss_cov + cov_weight * (diff * diff).sum()

            if use_center:
                loss_center = loss_center + 0.5 * ((feat_syn - prototypes[c]) ** 2).sum(1).mean()

        # Distribution-matching part only. This is the quantity the non-GAN arm
        # selects on, so keeping it separate is what makes the two arms
        # comparable.
        loss_dm = loss_mean + loss_cov
        if use_center:
            loss_dm = loss_dm + center_weight * loss_center

        loss_total = loss_dm
        if critic_active:
            critic.eval()
            loss_total = loss_total + adv_weight * (-critic(syn_data, label_syn).mean())

        # Selection scalar. With clean_selection the adversarial term is excluded:
        # a WGAN critic's output is unbounded and drifts in sign and scale as it
        # trains (the gradient penalty bounds the norm of its gradient, not its
        # value), so including it makes best_loss a non-stationary criterion and
        # best_syn can freeze on an early iterate for reasons unrelated to the
        # synthetic data. Default False reproduces the published behaviour.
        clean_selection = bool(config.get("dm_critic_clean_selection", False))
        sel_loss = loss_dm if clean_selection else loss_total
        cur = float((sel_loss / num_classes).detach().item())
        if np.isfinite(cur) and cur < best_loss:
            best_loss = cur
            best_it = it
            best_syn = syn_data.detach().clone()

        if torch.isfinite(loss_total):
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

        if snapshot_every and it % snapshot_every == 0:
            snapshots.append((it, syn_data.detach().clone()))

        if it % 100 == 0:
            extras = []
            if use_center:
                extras.append(f"center {loss_center.item() / num_classes:.6f}")
            if critic_active:
                extras.append("critic ON")
            ext = " | ".join(extras)
            print(f"[TAME-Critic] iter {it:04d} | loss {cur:.6f} | best {best_loss:.6f}@{best_it:04d}"
                  + (f" | {ext}" if ext else ""))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {"X_syn": best_syn.cpu(), "y_syn": label_syn.cpu(),
             "best_loss": best_loss, "best_it": best_it},
            os.path.join(save_dir, "best_syn.pt"),
        )

    if config.get("return_snapshots", False):
        return best_syn, label_syn.detach(), snapshots
    return best_syn, label_syn.detach()
