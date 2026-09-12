"""
TAME: Tabular Alignment via Moment Embeddings

Core synthesizer — mean + full covariance matching in frozen embedder space.
One random embedder per iteration, best-loss checkpoint, saves .pt output.
"""

import os
import torch
import numpy as np
from models.embedders import sample_random_embedder


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

    # Precompute row indices per class once. The optimisation is class-wise, so
    # every iteration needs to draw from a single class at a time; doing the
    # lookup here keeps it out of the hot loop
    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        """Draw n rows of class c. Samples with replacement only when the class
        is smaller than the request, which matters for imbalanced datasets."""
        idx = indices_class[c]
        return X_train[np.random.choice(idx, n, replace=len(idx) < n)]

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

    # Optional trail of intermediate states, used by the validation-based
    # snapshot selection
    snapshot_every = int(config.get("snapshot_every", 0))
    snapshots = []
    if snapshot_every:
        snapshots.append((-1, syn_data.detach().clone()))

    # Running best. Note this tracks the LOWEST LOSS, not the last state; the
    # final iterate is discarded unless it happens to be the best
    best_loss = float("inf")
    best_it = -1
    best_syn = syn_data.detach().clone()

    # --- MAIN LOOP ---
    for it in range(iters + 1):
        optimizer.zero_grad(set_to_none=True)   # clear last step's gradients

        # Accumulated separately (each term stays inspectable), but they are
        # summed before backward
        loss_mean = torch.zeros((), device=device)
        loss_cov = torch.zeros((), device=device)

        # --- VIEWS (embedders) LOOP ---
        # Create $views$ embedderss over the same iteration and comp the loss
        for _ in range(views):
            # A brand-new random network (embedder) every iteration, built frozen and in
            # eval mode. It is never trained and never reused. Matching moments
            # under many arbitrary projections is what stops syn_data from
            # overfitting to one particular embedding geometry
            embed_net = sample_random_embedder(
                embedder_type, embedder_size, input_dim, embed_hidden, embed_dim, device
            )
            embed_net.eval()

            # --- MEASURES LOOP ----
            for c in range(num_classes):
                # Take a real batch
                real_b = get_real_batch(c, batch_real)
                # Take synthetic batch
                syn_b = syn_data[c * ipc:(c + 1) * ipc]

                # Both sides go through the SAME embedder in the SAME iteration;
                # (comparing moments under different projections would be
                # dumb/meaningless)
                feat_real = embed_net(real_b).detach() # freeze real side into 
                                                       # a constant target in such
                                                       # way only the synthetic 
                                                       # path carries gradient
                feat_syn = embed_net(syn_b)

                mu_r, cov_r = cov_matrix(feat_real, eps)
                mu_s, cov_s = cov_matrix(feat_syn, eps)

                # Eq. 7. 
                # First term aligns position (squared L2 between
                # centroids)
                # Second aligns shape (squared Frobenius between
                # covariances, i.e. the sum over every matrix entry, so
                # off-diagonal correlations count too)
                #                       ‖μᵀ_c − μˢ_c‖²₂
                loss_mean = loss_mean + ((mu_r - mu_s) ** 2).sum()
                diff = cov_r - cov_s
                loss_cov = loss_cov + cov_weight * (diff * diff).sum()
                #                         λ         ‖Σᵀ_c − Σˢ_c‖²_F

        # This thing is a scalar tensor (a tensor with shape (), no dimensions)
        # basically a number but covered as a tensor, it is conected to the
        # computation graph and can propagate gradients
        loss_total = (loss_mean + loss_cov) / views

        # Per-class loss (for logging and checkpoint selection) 
        # IMPORTANT NOTE:
        # Values from different iterations are measured under different embedders and
        # are therefore not strictly comparable, which makes this a noisy
        # criterion; it is why the printed loss does not decrease monotonically
        cur = float((loss_total / num_classes).detach().item())
        if np.isfinite(cur) and cur < best_loss:
            best_loss = cur
            best_it = it
            best_syn = syn_data.detach().clone()   # clone: syn_data keeps moving

        # A non-finite loss skips the step instead of raising. Keeps a diverging
        # run alive, but it will produce nothing while looking healthy
        if torch.isfinite(loss_total):
            # backward() walks back THROUGH the frozen embedder to reach syn_data
            loss_total.backward() # comp ∂loss/∂syn_data and save in syn_data.grad
            torch.nn.utils.clip_grad_norm_([syn_data], grad_clip) # scale if
                                                                  # too big
            optimizer.step()        # the rows move here (syn_data -= lr * grad)

        if snapshot_every and it % snapshot_every == 0:
            snapshots.append((it, syn_data.detach().clone()))

        if it % 100 == 0:
            print(
                f"[TAME] iter {it:04d} | "
                f"loss {cur:.6f} | best {best_loss:.6f}@{best_it:04d}"
            )

    # Optional side channel; main.py does its own saving and leaves this unset.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {"X_syn": best_syn.cpu(), "y_syn": label_syn.cpu(),
             "best_loss": best_loss, "best_it": best_it},
            os.path.join(save_dir, "best_syn.pt"),
        )

    # Returns best_syn, NOT syn_data: the state after the final iteration is
    # thrown away. Callers branch on tuple length, so the arity must stay tied
    # to return_snapshots.
    if config.get("return_snapshots", False):
        return best_syn, label_syn.detach(), snapshots
    return best_syn, label_syn.detach()
