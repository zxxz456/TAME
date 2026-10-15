"""Snap every value of a distilled set to a value its column actually takes.

The multi-class diagnosis (`scripts/run_multi_diag.py`) found that tree accuracy
drops at the first SGD step, before the points have moved: that step makes every
value distinct, and in columns that take few values (letter's integers 0-15, the
zeros of spambase) a tree can then split between points that were equal in the
real data, on noise. Snapping the returned set back to observed values recovered
letter completely, but applied after distillation it also undid what TAME had
gained on credit. This is the in-loop version, built like the one-hot ``ste``:
the embedder only ever sees values that exist, the gradient passes straight
through to the continuous variable, and the returned set is snapped.

The grid of a column is every value it takes in the training data, so there is
no threshold to choose between "discrete" and "continuous". In a column with a
few values the snap restores the ties; in a truly continuous one the grid is so
dense that the snap moves values by almost nothing.
"""
from __future__ import annotations
import torch


class GridSnapper:
    """Nearest observed value, per column.

    Parameters
    ----------
    X_train : torch.Tensor
        ``(n, d)`` training matrix, already scaled. The grids come from here
    cols : torch.Tensor, optional
        Columns to snap. Defaults to all of them; the loop passes the columns
        outside the one-hot groups, which belong to the one-hot modes

    Attributes
    ----------
    cols : torch.Tensor
        The snapped columns
    sizes : torch.Tensor
        How many distinct values each of them has, for reporting
    """

    def __init__(self, X_train, cols=None):
        device = X_train.device
        d = X_train.shape[1]
        self.cols = (torch.arange(d, device=device) if cols is None else cols.to(device)).long()
        grids = [torch.unique(X_train[:, j]) for j in self.cols.tolist()]
        self.sizes = torch.tensor([len(g) for g in grids], device=device)
        # One padded (k, L) table instead of k separate ones, so a single
        # searchsorted handles every column. +inf keeps each row sorted and is
        # never the nearest value. L is at least 2 so a one-value column still
        # has a right neighbour to compare against.
        L = max(2, int(self.sizes.max())) if len(grids) else 2
        self.grid = torch.full((len(grids), L), float("inf"), device=device)
        for i, g in enumerate(grids):
            self.grid[i, :len(g)] = g

    def __bool__(self):
        return len(self.cols) > 0

    @torch.no_grad()
    def hard(self, X):
        """X with the chosen columns replaced by their nearest observed value."""
        if not self:
            return X
        V = X[:, self.cols].T.contiguous()                         # (k, n)
        i = torch.searchsorted(self.grid, V)                       # first index >= v
        i = i.clamp(min=1)
        i = torch.minimum(i, (self.sizes - 1).clamp(min=1).unsqueeze(1))
        izq = self.grid.gather(1, i - 1)
        der = self.grid.gather(1, i)
        near = torch.where((V - izq).abs() <= (der - V).abs(), izq, der)
        out = X.clone()
        out[:, self.cols] = near.T
        return out

    def ste(self, X):
        """Snapped forwards, identity backwards: X + (hard(X) - X).detach()."""
        if not self:
            return X
        return X + (self.hard(X) - X).detach()

    @torch.no_grad()
    def on_grid(self, X):
        """Share of the chosen entries that are exactly an observed value."""
        if not self:
            return 1.0
        return float((self.hard(X)[:, self.cols] == X[:, self.cols]).float().mean())
