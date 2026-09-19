"""
TAME: Tabular Alignment via Moment Embeddings

Core synthesizer — mean + full covariance matching in frozen embedder space.
One random embedder per iteration, best-loss checkpoint, saves .pt output.
"""

import os
import torch
import numpy as np
from models.embedders import sample_random_embedder, reinit_embedder_
from .instrument import IterTimer
from .onehot import OneHotProjector


def cov_matrix(z, eps=0.0):
    """First and second moments of a batch of embedded features

    Parameters
    ----------
    z : Tensor, shape (n, p)
        Embedded features. Rows are samples, columns are embedding dimensions
    eps : float, default 0.0
        Ridge added to the covariance diagonal

    Returns
    -------
    mu : Tensor, shape (p,)
        Centroid of the cloud
    cov : Tensor, shape (p, p)
        Covariance. The diagonal is the spread along each dimension

    """
    n = z.shape[0]
    mu = z.mean(0, keepdim=True)                 # (1, p), kept 2-D to broadcast
    zc = z - mu                                  # (n, p) centred
    # Outer product accumulated over samples: (p, n) @ (n, p) -> (p, p)
    # max(n, 1) only guards against an empty batch
    cov = (zc.T @ zc) / max(n, 1)
    if eps > 0:
        cov = cov + eps * torch.eye(cov.shape[0], device=z.device)
    return mu.squeeze(0), cov                    # (p,) and (p, p)


def tame_synthesize(data, config):
    """Distil a training set into a small synthetic set by matching per-class
    first- and second-order moments in randomly resampled embedder spaces

    The synthetic set is the optimisation variable: it is a leaf tensor with
    ``requires_grad=True`` that SGD updates directly. No downstream model is
    ever trained inside this function, which is what removes the inner loop of
    meta-model formulations

    Each iteration draws a fresh frozen embedder, projects a real minibatch and
    the synthetic set of every class through it, and penalises the discrepancy
    between the two projected clouds:

        L_c = ||mu_real - mu_syn||_2^2 + lambda * ||Sigma_real - Sigma_syn||_F^2

    Resampling the embedder every iteration is the point of the method: it
    approximates the expectation over the embedder distribution by Monte Carlo
    with one draw per step, so the synthetic set cannot overfit to any single
    projection

    Parameters
    ----------
    data : dict
        Output of ``prepare_db``. Requires ``X_train``, ``y_train``,
        ``input_dim`` and ``num_classes``.
    config : dict
        Required keys:
            device            : "cuda" or "cpu"
            ipc               : synthetic instances per class; the returned set
                                has ``ipc * num_classes`` rows
            dm_iters          : optimisation steps (the loop runs iters + 1)
            dm_lr             : SGD learning rate applied to the data
            dm_batch_real     : real samples drawn per class per iteration
            dm_embed_hidden   : embedder hidden width
            dm_embed_dim      : embedder output dim; must stay below ipc, or the
                                synthetic covariance is rank deficient and can
                                never match a full-rank target
            dm_embedder_type  : "ln_res_l", "dcnv2_base" or "node"
        Optional keys, with defaults:
            dm_embedder_size  : "base"  size rung in SIZE_LADDERS
            cov_weight        : 1.0     lambda in the loss above; 0.0 reduces
                                        the objective to mean-only matching
            grad_clip         : 10.0    max grad norm on the synthetic data
            moment_eps        : 1e-6    ridge added to both covariances
            dm_views          : 1       embedders averaged per iteration
            init_seed         : None    seeds the initial subset independently
                                        of the optimisation RNG stream, making
                                        the starting point reproducible across
                                        runs
            snapshot_every    : 0       if non-zero, keep a copy of the
                                        synthetic set every N iterations
            return_snapshots  : False   include that trail in the return value
            save_dir          : None    also write best_syn.pt there

    Returns
    -------
    (X_syn, y_syn) or (X_syn, y_syn, snapshots)
        ``X_syn`` is ``(ipc * num_classes, input_dim)`` and ``y_syn`` holds
        ``ipc`` copies of each class label, in order. The third element is
        returned only when ``return_snapshots`` is set, as a list of
        ``(iteration, tensor)`` pairs; iteration ``-1`` is the initialisation
    """
    # --- Preparation and config ---
    device = config["device"]
    X_train = data["X_train"].to(device).float()
    y_train = data["y_train"].to(device).long()

    # --- reqired config: no defaults, a missing key is a hard error ---
    ipc = int(config["ipc"])
    iters = int(config["dm_iters"])
    lr = float(config["dm_lr"])
    batch_real = int(config["dm_batch_real"])
    input_dim = int(data["input_dim"])          # from data
    num_classes = int(data["num_classes"])      # from data

    embed_hidden = int(config["dm_embed_hidden"])
    embed_dim = int(config["dm_embed_dim"])     # below ipc
    embedder_type = config["dm_embedder_type"]
    embedder_size = config.get("dm_embedder_size", "base")

    # --- optional conf ---
    grad_clip = float(config.get("grad_clip", 10.0))
    eps = float(config.get("moment_eps", 1e-6))
    cov_weight = float(config.get("cov_weight", 1.0))   # lambda in Eq. 7

    save_dir = config.get("save_dir", None)

    # --- one-hot inside the loop. Off by default; see synth/onehot.py ---
    # The dummies are two-valued z-scores after StandardScaler and the loop
    # moves them as free reals, which is what run_snap_projection.py showed
    # costs the tree classifiers. These modes keep them legal DURING the run:
    #   "none"     the published behaviour
    #   "ste"      the embedder sees hard one-hot rows; the gradient passes
    #              straight through to the continuous variable
    #   "soft"     softmax per group with a temperature annealed tau0 -> tau1
    #   "project"  hard projection after every SGD step (projected gradient)
    #   "penalty"  an L1 distance to the nearest legal value added to the loss,
    #              weight gamma. Soft at the default, converges to a hard
    #              projection as gamma grows
    onehot_mode = str(config.get("dm_onehot_mode", "none")).lower()
    if onehot_mode not in ("none", "ste", "soft", "project", "penalty"):
        raise ValueError(
            f"dm_onehot_mode={onehot_mode!r}; expected none, ste, soft, project or penalty")
    proj = OneHotProjector(X_train) if onehot_mode != "none" else None
    if proj is not None and not proj:
        # No one-hot groups in this dataset: nothing to project, so the mode
        # degrades to the published loop instead of pretending.
        onehot_mode, proj = "none", None
    tau0 = float(config.get("dm_onehot_tau0", 1.0))
    tau1 = float(config.get("dm_onehot_tau1", 0.05))
    # 100 is where the L1 term is measurable next to the moment gradient on
    # adult without yet clamping everything; 300 clamps 100% of the entries.
    onehot_gamma = float(config.get("dm_onehot_gamma", 100.0))

    def vista(x, it):
        """The rows the embedder sees this iteration."""
        if onehot_mode == "ste":
            return proj.ste(x)
        if onehot_mode == "soft":
            tau = tau0 * (tau1 / tau0) ** (it / max(iters, 1))
            return proj.soft(x, tau)
        return x

    def salida(x):
        """The rows that leave the loop: legal for the discrete modes, raw otherwise."""
        if onehot_mode in ("ste", "soft", "project"):
            return proj.hard(x)
        return x

    # Precompute row indices per class once. The optimisation is class-wise, so
    # every iteration needs to draw from a single class at a time; doing the
    # lookup here keeps it out of the hot loop
    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    cls_n = [len(i) for i in indices_class]

    def draw_rows(c, n):
        """Row ids for n samples of class c. With replacement only when the
        class is smaller than the request, which matters for imbalanced sets."""
        return np.random.choice(indices_class[c], n, replace=cls_n[c] < n)

    def get_real_batch(c, n):
        return X_train[draw_rows(c, n)]

    def get_real_all(n):
        """One batch of n rows for EVERY class, as (num_classes, n, input_dim).

        The draws are the same ones get_real_batch would make, class by class
        and in the same order, so the sampling is unchanged. What changes is
        that they reach the GPU as a single gather instead of one per class,
        which is what the per-class loop used to cost."""
        idx = np.concatenate([draw_rows(c, n) for c in range(num_classes)])
        return X_train[torch.as_tensor(idx, device=device)].view(num_classes, n, input_dim)

    # THE synthetic set: this is the optimisation variable
    # requires_grad=True makes the rows behave like model weights
    # Shape (ipc * num_classes, input_dim)
    syn_data = torch.randn((num_classes * ipc, input_dim), device=device, requires_grad=True)

    # Labels are assigned, never optimised: ipc copies of each class, in order
    # This balance the distilled set even when the source is skewed
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc)

    # Start from real rows rather than noise, in such way the starting point already
    # scores like a random baseline and the loop only has to improve on it...
    # A fixed init_seed decouples the starting subset from the optimisation RNG
    # stream, so several runs can share an initialisation and differ only in the
    # path taken from it
    init_seed = config.get("init_seed", None)
    with torch.no_grad():                       # an assignment, not a graph op
        if init_seed is not None:
            init_rng = np.random.default_rng(int(init_seed))
            for c in range(num_classes):
                idx = indices_class[c]
                take = init_rng.choice(idx, ipc, replace=len(idx) < ipc)
                syn_data[c * ipc:(c + 1) * ipc] = X_train[take]
        else:
            for c in range(num_classes):
                syn_data[c * ipc:(c + 1) * ipc] = get_real_batch(c, ipc)

    # The parameter list is the data itself, there is no model to update...
    optimizer = torch.optim.SGD([syn_data], lr=lr, momentum=0.5)

    # Embedders drawn per iteration. Losses are averaged over views, raising
    # this reduces gradient varianse without rescaling the effective lr
    views = int(config.get("dm_views", 1))

    # ONE embedder, redrawn in place every view. Constructing a module and
    # allocating its parameters is Python-side work that dominated the wall
    # clock for tensors this small; reinit_embedder_ gives a draw from the same
    # distribution without paying for it iters * views times.
    embed_net = sample_random_embedder(
        embedder_type, embedder_size, input_dim, embed_hidden, embed_dim, device,
        overrides=config.get("dm_embedder_overrides"),
    )
    embed_net.eval()

    # Optional trail of intermediate states, used by the validation-based
    # snapshot selection
    snapshot_every = int(config.get("snapshot_every", 0))
    snapshots = []
    if snapshot_every:
        snapshots.append((-1, salida(syn_data.detach()).clone()))

    # Running best. Note this tracks the LOWEST LOSS, not the last state; the
    # final iterate is discarded unless it happens to be the best
    # All three live on the device. Comparing a loss on the host would mean a
    # .item() every iteration, and each of those blocks until the GPU has
    # drained its queue; torch.where keeps the decision on the device, so the
    # queue never empties. They are read once, after the loop.
    best_loss = torch.full((), float("inf"), device=device)
    best_it = torch.full((), -1.0, device=device)
    best_syn = salida(syn_data.detach()).clone()

    # Per-iteration trace, also accumulated on device: (loss, mean part, cov
    # part, grad norm) per row, plus the two terms broken down by class. Both
    # are cheap, (iters+1) x 4 and (iters+1) x C x 2 floats.
    trace_t = torch.zeros((iters + 1, 4), device=device)

    # Per-iteration timer; see synth/instrument.py. It does not synchronise
    # inside the loop, which is exactly what this optimisation set out to avoid.
    timer = IterTimer(iters, device)
    per_class = torch.zeros((iters + 1, num_classes, 2), device=device)

    # --- MAIN LOOP ---
    for it in range(iters + 1):
        timer.tick(it)
        optimizer.zero_grad(set_to_none=True)   # clear last step's gradients

        # Accumulated separately (each term stays inspectable), but they are
        # summed before backward
        loss_mean = torch.zeros((), device=device)
        loss_cov = torch.zeros((), device=device)

        # --- VIEWS (embedders) LOOP ---
        # $views$ embedders over the same iteration, losses averaged
        for _ in range(views):
            # A fresh random draw of every weight, in place. The network is
            # never trained and never reused across views: matching moments
            # under many arbitrary projections is what stops syn_data from
            # overfitting to one particular embedding geometry.
            reinit_embedder_(embed_net)

            # --- MEASURES, ALL CLASSES AT ONCE ---
            # The classes used to be looped one by one, which meant num_classes
            # separate forwards, gathers and reductions per view. They are
            # independent, so they stack into a leading dimension and go through
            # the embedder as a single batch. Same arithmetic, one launch.
            real_b = get_real_all(batch_real)             # (C, B, d)
            syn_b = vista(syn_data, it).view(num_classes, ipc, input_dim)

            # Both sides go through the SAME embedder in the SAME iteration;
            # comparing moments under different projections would be meaningless
            feat_real = embed_net(                        # (C, B, p)
                real_b.reshape(-1, input_dim)
            ).detach().view(num_classes, batch_real, embed_dim)
            feat_syn = embed_net(                         # (C, ipc, p)
                syn_b.reshape(-1, input_dim)
            ).view(num_classes, ipc, embed_dim)

            # cov_matrix, batched over the class dimension. The real side is
            # detached above, so only the synthetic path carries gradient.
            mu_r = feat_real.mean(1)                      # (C, p)
            mu_s = feat_syn.mean(1)
            zr = feat_real - mu_r.unsqueeze(1)            # centred
            zs = feat_syn - mu_s.unsqueeze(1)
            cov_r = zr.transpose(1, 2) @ zr / max(batch_real, 1)   # (C, p, p)
            cov_s = zs.transpose(1, 2) @ zs / max(ipc, 1)
            if eps > 0:
                ridge = eps * torch.eye(embed_dim, device=device)
                cov_r = cov_r + ridge
                cov_s = cov_s + ridge

            # Eq. 7, per class and then summed.
            # First term aligns position (squared L2 between centroids)
            # Second aligns shape (squared Frobenius between covariances, i.e.
            # the sum over every matrix entry, so off-diagonal correlations
            # count too)
            #                     ‖μᵀ_c − μˢ_c‖²₂
            term_mean = ((mu_r - mu_s) ** 2).sum(-1)              # (C,)
            diff = cov_r - cov_s
            term_cov = cov_weight * (diff * diff).sum((-1, -2))   # (C,)
            #                         λ         ‖Σᵀ_c − Σˢ_c‖²_F
            loss_mean = loss_mean + term_mean.sum()
            loss_cov = loss_cov + term_cov.sum()

            # Which classes are hard, and whether it is position or shape that
            # is failing. Averaged over views at the end.
            with torch.no_grad():
                per_class[it, :, 0] += term_mean
                per_class[it, :, 1] += term_cov

        # This thing is a scalar tensor (a tensor with shape (), no dimensions)
        # basically a number but covered as a tensor, it is conected to the
        # computation graph and can propagate gradients
        loss_total = (loss_mean + loss_cov) / views
        if onehot_mode == "penalty":
            # Stationary and deterministic in syn_data, so unlike the critic
            # term it is safe to let it take part in the best-loss selection.
            pozo, suma = proj.penalty(syn_data)
            loss_total = loss_total + onehot_gamma * (pozo + suma)

        # Per-class loss, kept on the device.
        # IMPORTANT NOTE:
        # Values from different iterations are measured under different embedders
        # and are therefore not strictly comparable, which makes this a noisy
        # criterion; it is why the printed loss does not decrease monotonically
        cur = (loss_total / num_classes).detach()

        # Running best, decided on the device. torch.where selects without a
        # host-side branch, so nothing here forces the GPU queue to drain.
        improved = torch.isfinite(cur) & (cur < best_loss)
        best_loss = torch.where(improved, cur, best_loss)
        best_it = torch.where(improved, torch.as_tensor(float(it), device=device), best_it)
        best_syn = torch.where(improved, salida(syn_data.detach()), best_syn)

        # backward() walks back THROUGH the frozen embedder to reach syn_data.
        # A non-finite loss used to skip the step; zeroing the gradient instead
        # keeps that decision on the device. On a healthy run nan_to_num_ is a
        # no-op, and a diverging one produces nothing either way.
        loss_total.backward()   # comp ∂loss/∂syn_data and save in syn_data.grad
        torch.nan_to_num_(syn_data.grad, nan=0.0, posinf=0.0, neginf=0.0)
        # clip_grad_norm_ returns the norm it measured BEFORE clipping, which is
        # the honest size of the step the distillation wanted to take.
        grad_norm = torch.nn.utils.clip_grad_norm_([syn_data], grad_clip)
        optimizer.step()        # the rows move here (syn_data -= lr * grad)
        if onehot_mode == "project":
            with torch.no_grad():
                syn_data.copy_(proj.hard(syn_data))

        trace_t[it, 0] = cur
        trace_t[it, 1] = loss_mean.detach() / (views * num_classes)
        trace_t[it, 2] = loss_cov.detach() / (views * num_classes)
        trace_t[it, 3] = grad_norm

        if snapshot_every and it % snapshot_every == 0:
            snapshots.append((it, salida(syn_data.detach()).clone()))

        # The only sync in the loop, and it happens 11 times out of 1001.
        if it % 100 == 0:
            print(
                f"[TAME] iter {it:04d} | "
                f"loss {cur.item():.6f} | best {best_loss.item():.6f}@{int(best_it.item()):04d}"
            )

    # Off the device once, after the loop.
    dt_ms = timer.result()["dt_ms"]
    best_loss = float(best_loss.item())
    best_it = int(best_it.item())
    trace = [(i, *row, ms) for (i, row), ms
             in zip(enumerate(trace_t.cpu().tolist()), dt_ms)]

    # Optional side channel; main.py does its own saving and leaves this unset.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {"X_syn": best_syn.cpu(), "y_syn": label_syn.cpu(),
             "best_loss": best_loss, "best_it": best_it,
             "onehot_mode": onehot_mode,
             "onehot_drift": proj.drift(best_syn) if proj is not None else None,
             # (it, loss, loss_mean, loss_cov, grad_norm, dt_ms) per iteration
             "trace": trace,
             # (iters+1, num_classes, 2): the two terms, per class, per iteration
             "per_class": (per_class / views).cpu(),
             "snapshots": [(i, x.cpu()) for i, x in snapshots]},
            os.path.join(save_dir, "best_syn.pt"),
        )

    # Returns best_syn, NOT syn_data: the state after the final iteration is
    # thrown away. Callers branch on tuple length, so the arity must stay tied
    # to return_snapshots.
    if config.get("return_snapshots", False):
        return best_syn, label_syn.detach(), snapshots
    return best_syn, label_syn.detach()
