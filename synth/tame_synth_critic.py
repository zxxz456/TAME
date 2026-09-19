"""TAME plus an adversarial critic: the Sec. 2.7 / Table 11 variant.

Base TAME matches moments in embedding space, which says nothing about whether a
synthetic row is plausible as a row. This adds a WGAN-GP critic that scores rows
in *data* space and pulls the synthetic set toward regions it judges realistic,
and an optional center loss that pulls each class toward a learned prototype.

Registered as "tame_critic" in synth/registry.py.

Config keys beyond base TAME, all optional and all off or neutral by default:
    dm_use_center              False   enable the center loss
    dm_center_weight           0.1     its weight
    dm_use_critic              False   enable the adversarial term
    dm_adv_weight              0.05    lambda_adv in Eq. 13
    dm_n_critic                3       critic steps per synthetic step
    dm_critic_lr               1e-4    Adam lr for the critic
    dm_gp_lambda               10.0    gradient-penalty weight
    dm_critic_warmup           0       iterations before the critic engages
    dm_critic_clean_selection  False   see tame_critic_synthesize

Measured, on adult at IPC=50 with the published hyperparameters: the adversarial
term contributes about 1% of the loss value and 0.4% of the gradient on the
synthetic data. It is switched on but effectively inert, which is the simplest
explanation for Table 11 being inconclusive. Reaching gradient parity would need
dm_adv_weight near 12.8, some 250x the published value.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from models.embedders import sample_random_embedder, reinit_embedder_
from .instrument import IterTimer
from .tame_synth import cov_matrix


class CriticMLP(nn.Module):
    """Class-conditional WGAN critic over raw feature rows

    Scores a row as a single unbounded real number, not a probability: in a
    Wasserstein GAN the discriminator is a *critic*, so nothing squashes its
    output and it is free to drift in sign and scale while it trains. That drift
    is what motivates the clean_selection flag in tame_critic_synthesize

    Conditioning is by concatenation: the label goes through an embedding table
    and is appended to the features, so the critic learns a separate notion of
    "realistic" per class rather than one shared across the dataset.

    Parameters
    ----------
    input_dim : int
        Feature count. The critic works in data space, unlike the moment terms,
        which work in embedding space.
    num_classes : int
    hidden, depth : int
        Width and number of LeakyReLU blocks. Deliberately small: the critic is a
        regulariser, not the model under study.
    dropout : float
        Inserted between blocks when non-zero.
    """

    def __init__(self, input_dim, num_classes, hidden=256, depth=3, dropout=0.0):
        super().__init__()
        y_emb_dim = min(32, hidden)
        # embedding for the class labels
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
        # (B, d) + (B, y_emb_dim) -> (B,). squeeze drops the trailing 1 so the
        # score is a plain vector, one number per row.
        return self.net(torch.cat([x, self.y_emb(y)], dim=1)).squeeze(1)


def _gradient_penalty(D, x_real, y, x_fake, lam=10.0):
    """Two-sided gradient penalty, the GP of WGAN-GP.

    The Wasserstein objective is only valid for a 1-Lipschitz critic. This
    enforces it softly by penalising any deviation of the gradient norm from 1,
    measured at random points on the segments between real and synthetic rows

    Note what this does **not** bound: the critic's *output*. Only the norm of
    its gradient is constrained, so D(x) itself can and does drift, which is the
    mechanism behind the contaminated selection scalar documented below

    Parameters
    ----------
    D : nn.Module
        The critic, called as ``D(x, y)``.
    x_real, x_fake : Tensor, shape (b, d)
        Paired row-for-row; ``y`` labels both.
    lam : float
        Penalty weight.

    Returns
    -------
    Tensor
        Scalar, added to the critic loss.
    """
    b = x_real.size(0)
    # One interpolation coefficient per row, broadcast over features.
    eps = torch.rand(b, 1, device=x_real.device)
    x_hat = (eps * x_real + (1 - eps) * x_fake).requires_grad_(True)
    d_hat = D(x_hat, y)
    # create_graph=True keeps this differentiable: the penalty is part of the
    # critic's loss, so its gradient has to flow back into the critic weights.
    grads = torch.autograd.grad(d_hat.sum(), x_hat, create_graph=True, retain_graph=True)[0]
    return lam * ((grads.norm(2, dim=1) - 1) ** 2).mean()


def tame_critic_synthesize(data, config):
    """Distil with the moment objective plus an optional adversarial term.

    Same skeleton as ``tame_synthesize``: the synthetic set is the optimisation
    variable, a fresh frozen embedder is drawn every iteration, and per-class
    mean and covariance are matched in its space.

        L_total = L_DM + lambda_adv * (-E[D(s_syn, y)])

    with an optional center term folded into L_DM. Each iteration runs
    ``dm_n_critic`` critic steps first, then one step on the data.

    The two terms live in different spaces, which is the point: the moment terms
    see the embedder's projection, the critic sees the raw rows.

    Parameters
    ----------
    data : dict
        Output of ``prepare_db``; see ``tame_synthesize``.
    config : dict
        Same required keys as ``tame_synthesize``, plus the optional critic and
        center keys listed in the module docstring.

    Returns
    -------
    (X_syn, y_syn) or (X_syn, y_syn, snapshots)
        As in ``tame_synthesize``; the third element appears only when
        ``return_snapshots`` is set.

    """
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

    cls_n = [len(i) for i in indices_class]

    def draw_rows(c, n):
        return np.random.choice(indices_class[c], n, replace=cls_n[c] < n)

    def get_real_batch(c, n):
        return X_train[draw_rows(c, n)]

    def get_real_all(n):
        """One batch of n rows for EVERY class, as (num_classes, n, input_dim).

        Same draws as get_real_batch class by class, in the same order, so the
        sampling is unchanged; they just reach the GPU as one gather instead of
        num_classes separate ones."""
        idx = np.concatenate([draw_rows(c, n) for c in range(num_classes)])
        return X_train[torch.as_tensor(idx, device=device)].view(num_classes, n, input_dim)

    # Row ids per class, padded into one rectangular array so a whole critic
    # batch can be drawn with two vectorised numpy ops instead of batch_real
    # calls to np.random.choice. Costs C x max_class_size int64, a few MB.
    _cls_n = np.asarray(cls_n)
    _cls_rows = np.zeros((num_classes, int(_cls_n.max())), dtype=np.int64)
    for _c, _idx in enumerate(indices_class):
        _cls_rows[_c, :len(_idx)] = _idx

    def real_rows_for(labels):
        """One real row per entry of `labels`, drawn from that entry's class.

        The critic needs a batch paired class for class. Building it row by row
        meant batch_real draws plus batch_real index operations per critic step,
        three times an iteration. Here the whole batch is one uniform draw
        within each label's class, then a single gather."""
        pos = (np.random.rand(len(labels)) * _cls_n[labels]).astype(np.int64)
        return X_train[torch.as_tensor(_cls_rows[labels, pos], device=device)]

    syn_data = torch.randn((num_classes * ipc, input_dim), device=device, requires_grad=True)
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc)

    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c * ipc:(c + 1) * ipc] = get_real_batch(c, ipc)

    # Class prototypes for the center loss. They are learned, not fixed: they
    # join syn_data in the parameter list below and move with it
    prototypes = None
    if use_center:
        prototypes = torch.zeros((num_classes, embed_dim), device=device, requires_grad=True)

    # The optimiser's parameters are the data, and the prototypes when enabled.
    # No model weights are ever updated here.
    params = [syn_data] + ([prototypes] if use_center else [])
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.5)

    # Unlike the embedder, the critic persists across the whole run and is the
    # one component here that actually trains. beta1=0.5 is the usual GAN
    # setting: less momentum, so it tracks a moving opponent.
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
    # How often to probe the per-term gradient norms. 0 disables the probe.
    grad_log_every = int(config.get("grad_log_every", 0))
    snapshots = []
    if snapshot_every:
        snapshots.append((-1, syn_data.detach().clone()))   # the init subset

    # Every running best lives on the device. Comparing a loss on the host means
    # a .item() per iteration, and each one blocks until the GPU queue drains;
    # torch.where keeps the decision on the device so the queue never empties.
    inf = lambda: torch.full((), float("inf"), device=device)
    neg1 = lambda: torch.full((), -1.0, device=device)
    best_loss = inf()                    # the lowest loss encountered so far
    best_it = neg1()                     # the iteration at which the best loss
                                         # was found
    best_syn = syn_data.detach().clone() # the synthetic data corresponding to
                                         # the best loss

    # Two running bests from the SAME trajectory. The selection flag only decides
    # which one is returned, so tracking both here yields the contaminated and
    # the clean iterate out of a single run. That is stronger than running two
    # separate arms: those see different embedder sequences (the sampler seeds
    # itself from the wall clock), so part of any arm-to-arm difference is just a
    # different path rather than a different selection rule. Here the path is
    # identical and only the exit point differs.
    best_dm_loss, best_dm_it = inf(), neg1()
    best_dm_syn = syn_data.detach().clone()      # argmin of loss_dm alone
    best_tot_loss, best_tot_it = inf(), neg1()
    best_tot_syn = syn_data.detach().clone()     # argmin including the critic

    # Per-iteration trace of both candidate scalars, accumulated on device and
    # read once after the loop: (loss_dm, loss_total, gap, mean part, cov part,
    # g_dm, g_adv). NaN in the gradient columns on unprobed iterations.
    trace_t = torch.full((iters + 1, 7), float("nan"), device=device)

    # Per-iteration timer, with the critic block as a separate phase; see
    # synth/instrument.py. With n_critic critic steps per synthetic step,
    # splitting them says which of the two halves costs.
    timer = IterTimer(iters, device, spans=("critic",))

    # ONE embedder, redrawn in place each iteration. Building a module and
    # allocating its parameters is Python-side work that dominated the wall
    # clock; reinit_embedder_ draws from the same distribution without it.
    embed_net = sample_random_embedder(
        embedder_type, embedder_size, input_dim, embed_hidden, embed_dim, device,
        overrides=config.get("dm_embedder_overrides"),
    )
    embed_net.eval()

    for it in range(iters + 1):
        timer.tick(it)
        reinit_embedder_(embed_net)

        # --- critic update: n_critic steps before every step on the data ---
        # warmup delays engagement, so the critic is not random noise when it
        # first touches the synthetic set. Defaults to 0, i.e. no delay.
        critic_active = use_critic and it >= critic_warmup
        if critic_active:
            critic.train()
            for _ in range(n_critic):
                # delete grads of the critic before computing the new loss
                opt_critic.zero_grad(set_to_none=True)

                # Draw labels first, then one real and one synthetic row per
                # label, so the batch is paired class-for-class. Building the
                # real side row by row is the slow part of this variant: 128
                # separate draws per critic step.
                yb_np = np.random.randint(0, num_classes, batch_real)
                yb = torch.as_tensor(yb_np, device=device, dtype=torch.long)
                xb_real = real_rows_for(yb_np)
                idxs = torch.as_tensor(
                    yb_np * ipc + np.random.randint(0, ipc, batch_real),
                    device=device, dtype=torch.long)
                # detach: this step trains the critic, never the data.
                xb_syn = syn_data[idxs].detach()
                # Wasserstein objective, push D(syn) down, D(real) up.
                # The data update below pushes D(syn) the other way. With
                # n_critic=3 the critic normally wins the exchange.
                loss_D = critic(xb_syn, yb).mean() - critic(xb_real, yb).mean()
                loss_D = loss_D + _gradient_penalty(critic, xb_real, yb, xb_syn, gp_lambda)
                loss_D.backward()
                opt_critic.step()

        timer.mark("critic", it)       # the critic block ends here

        # syn update
        optimizer.zero_grad(set_to_none=True)

        loss_mean = torch.zeros((), device=device) # first term of ec7
        loss_cov = torch.zeros((), device=device)  # second term of ec7
        loss_center = torch.zeros((), device=device)

        # -- Distillation, all classes at once ---
        # The classes are independent, so instead of num_classes forwards,
        # gathers and reductions they stack into a leading dimension and cross
        # the embedder as one batch. Same arithmetic, one launch.
        real_b = get_real_all(batch_real)                 # (C, B, d)
        syn_b = syn_data.view(num_classes, ipc, input_dim)

        feat_real = embed_net(                            # (C, B, p)
            real_b.reshape(-1, input_dim)
        ).detach().view(num_classes, batch_real, embed_dim)
        feat_syn = embed_net(                             # (C, ipc, p)
            syn_b.reshape(-1, input_dim)
        ).view(num_classes, ipc, embed_dim)

        # cov_matrix, batched over the class dimension.
        mu_r, mu_s = feat_real.mean(1), feat_syn.mean(1)          # (C, p)
        zr = feat_real - mu_r.unsqueeze(1)
        zs = feat_syn - mu_s.unsqueeze(1)
        cov_r = zr.transpose(1, 2) @ zr / max(batch_real, 1)      # (C, p, p)
        cov_s = zs.transpose(1, 2) @ zs / max(ipc, 1)
        if eps > 0:
            ridge = eps * torch.eye(embed_dim, device=device)
            cov_r, cov_s = cov_r + ridge, cov_s + ridge

        loss_mean = loss_mean + ((mu_r - mu_s) ** 2).sum()
        diff = cov_r - cov_s
        loss_cov = loss_cov + cov_weight * (diff * diff).sum()

        if use_center:
            # Squared distance of each projected synthetic row to its class
            # prototype: compresses within-class spread.
            loss_center = loss_center + 0.5 * (
                (feat_syn - prototypes.unsqueeze(1)) ** 2).sum(-1).mean(-1).sum()

        # Distribution-matching part only. This is the quantity the non-GAN arm
        # selects on, so keeping it separate is what makes the two arms
        # comparable.
        loss_dm = loss_mean + loss_cov
        if use_center:
            loss_dm = loss_dm + center_weight * loss_center

        loss_total = loss_dm
        adv_term = None          # stays None when the critic is off
        if critic_active:
            # Ec 12. The critic is frozen here; it already had its turn above.
            # The minus sign inverts its objective: the data move toward rows the
            # critic scores as realistic. syn_data goes in whole rather than per
            # class, since the critic is conditional and takes (x, y) pairs.
            critic.eval()
            adv_term = adv_weight * (-critic(syn_data, label_syn).mean())
            loss_total = loss_total + adv_term

        # Selection scalar. With clean_selection the adversarial term is excluded:
        # a WGAN critic's output is unbounded and drifts in sign and scale as it
        # trains (the gradient penalty bounds the norm of its gradient, not its
        # value), so including it makes best_loss a non-stationary criterion and
        # best_syn can freeze on an early iterate for reasons unrelated to the
        # synthetic data. Default False reproduces the published behaviour.
        clean_selection = bool(config.get("dm_critic_clean_selection", False))
        sel_loss = loss_dm if clean_selection else loss_total
        it_t = torch.as_tensor(float(it), device=device)

        def keep_best(cur, best_loss, best_it, best_syn):
            """Running argmin, decided on the device so nothing syncs."""
            better = torch.isfinite(cur) & (cur < best_loss)
            return (torch.where(better, cur, best_loss),
                    torch.where(better, it_t, best_it),
                    torch.where(better, syn_data.detach(), best_syn))

        cur = (sel_loss / num_classes).detach()
        best_loss, best_it, best_syn = keep_best(cur, best_loss, best_it, best_syn)

        # Both candidate scalars, tracked regardless of which rule is active, so
        # one run produces both iterates and the full trace to compare them.
        cur_dm = (loss_dm / num_classes).detach()
        cur_tot = (loss_total / num_classes).detach()
        best_dm_loss, best_dm_it, best_dm_syn = keep_best(
            cur_dm, best_dm_loss, best_dm_it, best_dm_syn)
        best_tot_loss, best_tot_it, best_tot_syn = keep_best(
            cur_tot, best_tot_loss, best_tot_it, best_tot_syn)

        trace_t[it, 0] = cur_dm
        trace_t[it, 1] = cur_tot
        trace_t[it, 2] = cur_tot - cur_dm
        trace_t[it, 3] = loss_mean.detach() / num_classes
        trace_t[it, 4] = loss_cov.detach() / num_classes

        # Gradient magnitude of each term with respect to the synthetic data.
        # This is what actually decides whether the critic moves the rows: the
        # two terms can differ in value by 1% and in gradient by far more or far
        # less. Each probe costs one extra backward pass, hence the schedule;
        # unprobed iterations keep the NaN the buffer was filled with.
        if grad_log_every and it % grad_log_every == 0:
            trace_t[it, 5] = torch.autograd.grad(
                loss_dm, syn_data, retain_graph=True)[0].norm()
            trace_t[it, 6] = 0.0 if adv_term is None else torch.autograd.grad(
                adv_term, syn_data, retain_graph=True)[0].norm()

        # A non-finite loss used to skip the step; zeroing the gradient instead
        # keeps that decision on the device. On a healthy run nan_to_num_ is a
        # no-op, and a diverging one produces nothing either way.
        loss_total.backward()
        for q in params:
            torch.nan_to_num_(q.grad, nan=0.0, posinf=0.0, neginf=0.0)
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
            # The only sync in the loop, 11 times out of 1001.
            print(f"[TAME-Critic] iter {it:04d} | loss {cur.item():.6f} | "
                  f"best {best_loss.item():.6f}@{int(best_it.item()):04d}"
                  + (f" | {ext}" if ext else ""))

    # Off the device once, after the loop.
    crono = timer.result()
    dt_ms, dt_critic = crono["dt_ms"], crono["critic_ms"]
    best_loss, best_it = float(best_loss.item()), int(best_it.item())
    best_dm_loss, best_dm_it = float(best_dm_loss.item()), int(best_dm_it.item())
    best_tot_loss, best_tot_it = float(best_tot_loss.item()), int(best_tot_it.item())
    trace = [(i, *row, ms, mc) for (i, row), ms, mc
             in zip(enumerate(trace_t.cpu().tolist()), dt_ms, dt_critic)]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {"X_syn": best_syn.cpu(), "y_syn": label_syn.cpu(),
             "best_loss": best_loss, "best_it": best_it,
             # Both iterates from this same trajectory, so a caller can score the
             # two selection rules without distilling twice.
             "X_syn_dm": best_dm_syn.cpu(),
             "best_dm_loss": best_dm_loss, "best_dm_it": best_dm_it,
             "X_syn_total": best_tot_syn.cpu(),
             "best_total_loss": best_tot_loss, "best_total_it": best_tot_it,
             # (it, loss_dm, loss_total, gap, loss_mean, loss_cov, g_dm, g_adv,
             #  dt_ms, dt_critic_ms) por iteracion
             "trace": trace,
             # The synthetic set itself every snapshot_every iterations, so the
             # trajectory can be replayed offline instead of only its losses.
             "snapshots": [(i, x.cpu()) for i, x in snapshots]},
            os.path.join(save_dir, "best_syn.pt"),
        )

    if config.get("return_snapshots", False):
        return best_syn, label_syn.detach(), snapshots
    return best_syn, label_syn.detach()
