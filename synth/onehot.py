"""The one-hot projection, for use INSIDE the distillation loop.

`scripts/run_snap_projection.py` projects at evaluation time only: it distils as
usual and snaps the rows back just before the classifier is trained. That
measures whether the defect matters, but it does not fix it: the optimisation
keeps moving the dummies through ground that does not exist, and the mess is
only cleaned up at the end.

Here the projection enters the loop. The problem is that argmax is not
differentiable, so there is more than one reasonable way to do it and none of
them is the obvious one. This module implements four and lets the experiment
decide:

  ``hard``     the projection as such, with no gradient. It is what evaluation
               uses and what defines the set the loop returns.
  ``ste``      *straight-through estimator*: the embedder sees genuine one-hot
               rows and the gradient reaches the continuous variable as if the
               projection were the identity. This is the literal reading of
               "put the projection inside the loop": moments are compared
               between legal rows on both sides.
  ``soft``     softmax per group with a temperature annealed from ``tau0`` to
               ``tau1``. A high temperature gives a smooth gradient and a low
               one converges to argmax. It avoids the estimator's bias at the
               price of one more hyperparameter.
  ``penalty``  projects nothing: it adds to the loss an L1 distance from every
               dummy to its nearest legal value, plus a term pushing each
               group's sum to 0 or 1. This is the soft version, the one a
               reviewer will ask for as a baseline.

And ``project``, which does not live here but in the loop: take the SGD step and
then project the variable. That is textbook projected gradient descent, with no
estimator, but it freezes the categorical columns as soon as the step is no
longer large enough to change an argmax.
"""
from __future__ import annotations
import torch


@torch.no_grad()
def detect_groups(X: torch.Tensor) -> list[torch.Tensor]:
    """The one-hot groups, inferred from the data rather than from a list.

    Looks for runs of two-valued columns in which, row by row, at most one sits
    at its high value. On `adult` this recovers exactly the eight groups
    `pd.get_dummies` produced

    Parameters
    ----------
    X : torch.Tensor
        ``(n, d)`` training matrix, already scaled

    Returns
    -------
    list[torch.Tensor]
        Column indices of each group, including lone binary columns, which come
        out as groups of size one
    """
    Xc = X.detach().cpu()
    n, d = Xc.shape
    binaria = [torch.unique(Xc[:, j]).numel() == 2 for j in range(d)]
    alto = torch.stack([(Xc[:, j] == Xc[:, j].max()).float() if binaria[j]
                        else torch.zeros(n) for j in range(d)], 1)
    grupos, j = [], 0
    while j < d:
        if not binaria[j]:
            j += 1
            continue
        g, acc, k = [j], alto[:, j].clone(), j + 1
        while k < d and binaria[k] and ((acc + alto[:, k]) <= 1).all():
            acc += alto[:, k]
            g.append(k)
            k += 1
        grupos.append(torch.tensor(g))
        j = k
    return grupos


class OneHotProjector:
    """Returns the dummies of a synthetic row to their legal values.

    After the ``StandardScaler`` a dummy is no longer 0 and 1 but two z-scores,
    one per column. Both are stored, and the work is done on the *score*
    ``s = (x - z0) / (z1 - z0)``, which is exactly 0 or 1 in any real row and
    anything else in a distilled one

    Only groups of two or more columns are touched. A lone binary column has
    nobody to compete against in an argmax, so it is left alone: projecting it
    would be a different decision, not the same one

    Parameters
    ----------
    X_train : torch.Tensor
        ``(n, d)`` training matrix. The two values per column come from here
    groups : list[torch.Tensor], optional
        The groups. Inferred with ``detect_groups`` when not given
    device : str, optional
        Where to keep the tensors. Defaults to ``X_train``'s

    Attributes
    ----------
    groups : list of (Tensor, Tensor, Tensor)
        ``(indices, z0, z1)`` per group, all already on ``device``
    n_dummy : int
        How many columns it covers, so it can be reported
    """

    def __init__(self, X_train, groups=None, device=None):
        device = device or X_train.device
        groups = detect_groups(X_train) if groups is None else groups
        self.groups = []
        for g in groups:
            if len(g) < 2:
                continue
            g = g.to(device)
            cols = X_train[:, g]
            z0, z1 = cols.min(0).values, cols.max(0).values
            if bool((z1 <= z0).any()):
                continue                      # constant group: nothing to project
            self.groups.append((g, z0, z1))
        self.n_dummy = sum(int(g.numel()) for g, _, _ in self.groups)
        self.n_groups = len(self.groups)

        # --- the same groups, as rectangular tensors ---
        # Walking them in Python costs two kernel launches per group per call,
        # and inside the distillation loop that is `iters` x groups x 2. On
        # `splice`, with 61 groups, it made distillation forty times slower than
        # the control arm, which is a difference of implementation and not of
        # method. Padded to the width of the largest group, all the work fits in
        # one operation over (rows, groups, width)
        G, K = self.n_groups, (max(int(g.numel()) for g, _, _ in self.groups)
                               if self.n_groups else 0)
        cols = torch.zeros((G, K), dtype=torch.long, device=device)
        z0 = torch.zeros((G, K), device=device)
        z1 = torch.ones((G, K), device=device)
        valid = torch.zeros((G, K), dtype=torch.bool, device=device)
        for i, (g, a, b) in enumerate(self.groups):
            k = int(g.numel())
            cols[i, :k], z0[i, :k], z1[i, :k], valid[i, :k] = g, a, b, True
        self._cols, self._z0, self._z1 = cols, z0, z1
        self._valid = valid
        self._ancho = valid.sum(1, keepdim=True).clamp(min=1)   # live entries per group
        self._plano = cols[valid]                               # (n_dummy,) real indices
        self._rango = (z1 - z0).clamp(min=1e-12)

    def __bool__(self):
        return self.n_groups > 0

    def _score(self, X, g=None, z0=None, z1=None):
        """The score: 0 if the category is off, 1 if it is on.

        With a single group it returns that group; with no arguments, all of
        them at once shaped ``(rows, groups, width)``. Padding positions come
        out as zero and must be ignored through ``self._valid``
        """
        if g is not None:
            return (X[:, g] - z0) / (z1 - z0)
        return torch.where(self._valid, (X[:, self._cols] - self._z0) / self._rango,
                           X.new_zeros(()))

    def _escribir(self, X, out):
        """Writes the group columns back into ``X``, in one go."""
        X = X.clone()
        X[:, self._plano] = out[:, self._valid]
        return X

    @torch.no_grad()
    def hard(self, X):
        """The projection, with no gradient. Each group keeps its argmax.

        A group whose best score does not reach 0.5 stays all-zero, which is
        what a real row with a missing category looks like. That is what makes
        the projection the identity on real rows, and that is what makes it
        defensible
        """
        if not self.n_groups:
            return X.clone()
        s = self._score(X).masked_fill(~self._valid, float("-inf"))
        mejor, k = s.max(-1)                               # (rows, groups)
        out = self._z0.expand(X.shape[0], -1, -1).clone()
        alto = torch.gather(self._z1.expand_as(out), -1, k.unsqueeze(-1))
        # Only the winner is switched on, and only if it got at least halfway.
        out.scatter_(-1, k.unsqueeze(-1),
                     torch.where((mejor >= 0.5).unsqueeze(-1), alto,
                                 torch.gather(self._z0.expand_as(out), -1, k.unsqueeze(-1))))
        return self._escribir(X, out)

    def ste(self, X):
        """Hard projection forwards, identity backwards.

        ``X + (hard(X) - X).detach()`` equals ``hard(X)`` in the forward pass and
        has derivative one with respect to ``X``, so the embedder compares
        moments between legal rows while the gradient still reaches the
        continuous variable
        """
        return X + (self.hard(X) - X).detach()

    def soft(self, X, tau: float):
        """Softmax per group with a temperature, instead of the argmax.

        A large ``tau`` spreads the weight and keeps the gradient smooth; a small
        one approaches the argmax. Annealing ``tau`` over the run gives the best
        of both, at the price of one hyperparameter

        Unlike ``hard``, the softmax always sums to one per group, so it cannot
        represent an absent category
        """
        if not self.n_groups:
            return X.clone()
        s = self._score(X).masked_fill(~self._valid, float("-inf"))
        w = torch.softmax(s / max(tau, 1e-6), dim=-1)
        return self._escribir(X, self._z0 + w * (self._z1 - self._z0))

    def penalty(self, X):
        """How far outside the legal values it went, as a differentiable scalar.

        Two terms, both L1 distances to the nearest legal value. The first
        applies to each dummy's score, ``|s - round(clip(s))|``, and pushes it
        towards 0 or towards 1, whichever is closer, without deciding which. The
        second applies to the group's sum and is what stops three categories
        from switching on at once

        It is L1 and not a double well ``(s(1-s))^2`` for the same reason L1
        produces exact zeros and L2 does not: the double well's gradient
        vanishes as it approaches 0 and 1, so it gets close and never arrives,
        and on `adult` at 300 iterations it never passed 13% of legal entries at
        any weight. The L1 distance pulls with constant force until the value is
        pinned: at ``gamma=300`` it leaves 100% exact. The target being aimed at
        is ``detach``ed, so the gradient is the distance's and does not move the
        target

        The sum goes to 0 **or** to 1, not to 1 flat: a real row can have the
        whole group at zero when the category was absent, and pulling it towards
        one would be inventing data. In this form the penalty is exactly zero on
        real rows, which is the same condition ``hard`` satisfies

        It projects nothing: it is a soft constraint added to Eq. 7. With a high
        ``gamma`` it converges to the hard projection; with a moderate one it
        stops halfway, and that is where its interest as a baseline lies
        """
        if not self.n_groups:
            return X.new_zeros(()), X.new_zeros(())
        s = self._score(X)
        d = (s - s.detach().clamp(0, 1).round()).abs() * self._valid
        # Mean within the group and then across groups, like the per-group code
        # this replaces: with groups of different sizes that is not the same as
        # one flat mean.
        pozo = (d.sum(-1) / self._ancho.squeeze(-1)).mean()
        t = s.sum(-1)
        suma = (t - t.detach().clamp(0, 1).round()).abs().mean()
        return pozo, suma

    @torch.no_grad()
    def drift(self, X):
        """Diagnostic: mean distance to the legal value and fraction outside [0, 1].

        Useful to confirm the chosen mode is doing something. With ``ste`` or
        ``project`` the distance of the returned set has to be zero
        """
        if not self.n_groups:
            return 0.0, 0.0
        s = self._score(X)[:, self._valid].reshape(-1)
        return (float((s - s.clamp(0, 1).round()).abs().mean()),
                float(((s < 0) | (s > 1)).float().mean()))
