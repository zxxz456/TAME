"""Per-iteration timing for the synthesizers, without syncing the GPU.

Timing a CUDA loop naively means calling ``torch.cuda.synchronize()`` every
iteration, which forces the host to wait for the queue to drain and so changes
the very thing being measured. Recording a ``torch.cuda.Event`` instead is
asynchronous: the event is queued like any other kernel and its timestamp is
read once, after the loop.

The whole footprint inside a synthesizer is three lines:

    timer = IterTimer(iters, device)          # before the loop
    for it in range(iters + 1):
        timer.tick(it)                        # first line of the loop body
        ...
    dt = timer.result()["dt_ms"]              # after the loop

A synthesizer with an inner phase worth separating declares it as a span and
closes it where that phase ends:

    timer = IterTimer(iters, device, spans=("critic",))
    for it in range(iters + 1):
        timer.tick(it)
        ...critic steps...
        timer.mark("critic", it)
        ...the rest of the iteration...
    r = timer.result()                        # {"dt_ms": [...], "critic_ms": [...]}

On CPU it falls back to ``time.perf_counter``, which is exact there because
there is no queue to drain.
"""

import time

import torch


class IterTimer:
    """Wall-clock per iteration, measured without draining the CUDA queue.

    Parameters
    ----------
    iters : int
        Iterations the loop will run. The synthesizers loop over
        ``range(iters + 1)``, so ``iters + 1`` measurements come back.
    device : str or torch.device
        Anything not starting with "cuda" selects the ``perf_counter`` path.
    spans : tuple of str, optional
        Names of inner phases to time separately. Each span starts at ``tick``
        and ends at its own ``mark``.
    enabled : bool, default True
        False turns every method into a no-op and makes ``result`` return empty
        lists, so instrumentation can be switched off without branching at the
        call site.

    Notes
    -----
    The first iteration always reads high: it carries kernel compilation and
    allocator warm-up. Drop it before averaging.

    Only the loop is measured. Whatever the synthesizer does before it (building
    the embedder, initialising the synthetic set) and after it (saving the
    checkpoint) falls outside, so the sum of ``dt_ms`` comes out below a
    stopwatch wrapped around the whole call. That gap is the setup cost, and
    keeping it out is deliberate: it does not scale with ``iters``.
    """

    def __init__(self, iters, device, spans=(), enabled=True):
        self.iters = int(iters)
        self.spans = tuple(spans)
        self.enabled = bool(enabled)
        self.cuda = self.enabled and str(device).startswith("cuda")
        self._stopped = False

        # One boundary per iteration plus a closing one; each span needs its own
        # end marker per iteration, since it always starts at that tick.
        n = self.iters + 2
        if not self.enabled:
            self._bounds, self._marks = [], {}
        elif self.cuda:
            ev = lambda: torch.cuda.Event(enable_timing=True)
            self._bounds = [ev() for _ in range(n)]
            self._marks = {s: [ev() for _ in range(n - 1)] for s in self.spans}
        else:
            self._bounds = [0.0] * n
            self._marks = {s: [0.0] * (n - 1) for s in self.spans}

    def tick(self, it):
        """Open iteration ``it``. Goes as the first line of the loop body."""
        if not self.enabled:
            return
        if self.cuda:
            self._bounds[it].record()
        else:
            self._bounds[it] = time.perf_counter()

    def mark(self, span, it):
        """Close ``span`` for iteration ``it``, at the point that phase ends."""
        if not self.enabled:
            return
        if self.cuda:
            self._marks[span][it].record()
        else:
            self._marks[span][it] = time.perf_counter()

    def stop(self):
        """Close the last iteration. ``result`` calls it, so it is optional."""
        if not self.enabled or self._stopped:
            return
        if self.cuda:
            self._bounds[self.iters + 1].record()
            torch.cuda.synchronize()        # the one sync, and it is after the loop
        else:
            self._bounds[self.iters + 1] = time.perf_counter()
        self._stopped = True

    def result(self):
        """Milliseconds per iteration, plus one list per declared span.

        Returns
        -------
        dict
            ``{"dt_ms": [...]}`` with one entry per iteration, and
            ``"<span>_ms"`` for each span declared at construction. Empty lists
            when the timer is disabled.
        """
        if not self.enabled:
            return {"dt_ms": [], **{f"{s}_ms": [] for s in self.spans}}
        self.stop()
        rng = range(self.iters + 1)
        if self.cuda:
            out = {"dt_ms": [self._bounds[i].elapsed_time(self._bounds[i + 1]) for i in rng]}
            for s in self.spans:
                out[f"{s}_ms"] = [self._bounds[i].elapsed_time(self._marks[s][i]) for i in rng]
        else:
            out = {"dt_ms": [(self._bounds[i + 1] - self._bounds[i]) * 1e3 for i in rng]}
            for s in self.spans:
                out[f"{s}_ms"] = [(self._marks[s][i] - self._bounds[i]) * 1e3 for i in rng]
        return out
