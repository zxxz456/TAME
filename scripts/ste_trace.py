"""Step by step trace of the straight-through estimator, on a real distillation.

`ste` is one line, and the line is the confusing part: what leaves it forwards
and what comes back through it are not the derivative of the same function. This
records a real run of the loop and lays out what actually happened, iteration by
iteration, so the mechanism can be read off numbers instead of argued about.

Nothing here reimplements the loop. `OneHotProjector.ste` is wrapped at class
level for the duration of the run, so what gets recorded is what `tame_synth`
really passed to the embedder, and the gradient comes from a hook on the tensor
that came out, which by the estimator's own definition is the gradient that
lands on `syn_data`.

Reproducibility. `models/embedders.py` seeds the global RNG from the wall clock
on every embedder draw, so two identical runs do not match. The clock is frozen
while distilling, which makes the run a function of `--seed` alone, and the
recording is written to disk and reused. Delete the cache or pass `--force` to
run it again.

    .venv/bin/python scripts/ste_trace.py --grupo relationship --fila 0
"""
from __future__ import annotations
import os, sys, json, time, random, argparse
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.prepare_database import prepare_db
from synth.registry import synthesize
from synth.onehot import OneHotProjector
import models.embedders as _emb

SALIDA = os.path.expanduser("~/tame_runs/ste_trace")


def _semilla(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class _RelojQuieto:
    """Freezes `time.time` inside `models.embedders` for the duration of the block.

    `sample_random_embedder` reseeds the global RNG from the clock on every call,
    which is once per iteration, so the caller's seed is worth nothing. The file
    is not patched, it is the paper's code: its clock is stopped while distilling
    """
    def __init__(self, t=1_700_000_000.0):
        self.t, self.orig = t, None

    def __enter__(self):
        self.orig = _emb.time.time
        _emb.time.time = lambda: self.t
        return self

    def __exit__(self, *e):
        _emb.time.time = self.orig


class _Grabadora:
    """Records every call to ``ste`` and the gradient that comes back through it.

    Wrapping the method on the class, rather than passing a different object into
    the loop, is what keeps this honest: `tame_synth` builds its own projector
    and never learns it is being watched, so the numbers are the ones the run
    really used

    Three things are kept per iteration, restricted to one group's columns:
    the raw values of the optimisation variable, the values the embedder was
    handed, and the gradient. The gradient comes from a hook on the returned
    tensor; with a straight-through estimator that is exactly the gradient that
    reaches ``syn_data``, because the derivative in between is one
    """

    def __init__(self, cols):
        self.cols = cols
        self.crudo, self.visto, self.grad = [], [], []
        self._orig = None

    def __enter__(self):
        self._orig = OneHotProjector.ste
        rec = self

        def ste_grabado(proj_self, X):
            V = rec._orig(proj_self, X)
            rec.crudo.append(X.detach()[:, rec.cols].cpu().numpy().copy())
            rec.visto.append(V.detach()[:, rec.cols].cpu().numpy().copy())
            hueco = len(rec.crudo) - 1
            rec.grad.append(None)

            def guardar(g, k=hueco):
                rec.grad[k] = g.detach()[:, rec.cols].cpu().numpy().copy()
                return g

            if V.requires_grad:
                V.register_hook(guardar)
            return V

        OneHotProjector.ste = ste_grabado
        return self

    def __exit__(self, *e):
        OneHotProjector.ste = self._orig

    def arrays(self):
        """The three stacks as ``(iteraciones, filas, columnas del grupo)``."""
        n = len(self.crudo)
        g = [x if x is not None else np.full_like(self.crudo[0], np.nan) for x in self.grad]
        return (np.stack(self.crudo), np.stack(self.visto), np.stack(g[:n]))


class TrazaSTE:
    """One categorical group followed through a real `ste` run, step by step.

    Parameters
    ----------
    grupo : str
        Prefix of the categorical being followed, for example ``relationship``
    name : str
        Dataset. Only the ones `prepare_db` knows
    seed, ipc, iters, lr, embedder : ...
        The distillation. Defaults to the experiment's
    out : str
        Where the recording and the tables are written

    Attributes
    ----------
    crudo, visto, grad : numpy.ndarray
        ``(iteraciones, filas, ancho del grupo)``. What the variable held, what
        the embedder was handed, and what came back
    """

    def __init__(self, grupo="relationship", name="adult", seed=132, ipc=10,
                 iters=1000, lr=0.5, embedder="ln_res_l", out=SALIDA,
                 device=None, force=False):
        self.grupo_nom, self.name, self.seed = grupo, name, seed
        self.ipc, self.iters, self.lr, self.embedder = ipc, iters, lr, embedder
        self.out = out
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(out, exist_ok=True)

        self.data = prepare_db({"random_seed": seed, "device": self.dev}, name=name)
        self.X_train = self.data["X_train"]
        self.proj = OneHotProjector(self.X_train)
        self.cols, self.z0, self.z1 = self._buscar_grupo(grupo)
        self.nombres = self._nombres()

        self.crudo, self.visto, self.grad, self.X_syn, self.meta = self._correr(force)

    # ---- the group being followed ----

    def _buscar_grupo(self, prefijo):
        """The group whose columns all belong to one categorical.

        Groups come out of `detect_groups`, which reads them off the data, so
        they are runs of column indices with no names attached. Names are
        recovered separately and only to make the tables readable
        """
        nombres = self._nombres_tabla()
        for g, z0, z1 in self.proj.groups:
            idx = [int(j) for j in g]
            if nombres and all(nombres[j].startswith(prefijo + "_") for j in idx):
                return torch.tensor(idx), z0.cpu(), z1.cpu()
            if not nombres and prefijo == str(idx[0]):
                return torch.tensor(idx), z0.cpu(), z1.cpu()
        hay = sorted({nombres[int(g[0])].split("_")[0] for g, _, _ in self.proj.groups}) \
            if nombres else [str(int(g[0])) for g, _, _ in self.proj.groups]
        raise ValueError(f"no group is entirely {prefijo}. Available: {hay}")

    def _nombres_tabla(self):
        """Column names after get_dummies, or None when they cannot be recovered.

        `prepare_db` returns tensors and drops the names, so they are rebuilt
        from the raw frame. Purely cosmetic: everything else works without them
        """
        if getattr(self, "_cache_nombres", None) is not None:
            return self._cache_nombres
        self._cache_nombres = []
        try:
            from sklearn.datasets import fetch_openml
            fuentes = {"adult": ("adult", 2)}
            if self.name not in fuentes:
                return self._cache_nombres
            nom, ver = fuentes[self.name]
            df = fetch_openml(nom, version=ver, as_frame=True).data
            self._cache_nombres = list(pd.get_dummies(df, drop_first=False).columns)
        except Exception:
            pass
        return self._cache_nombres

    def _nombres(self):
        t = self._nombres_tabla()
        if not t:
            return [f"col{int(j)}" for j in self.cols]
        return [t[int(j)].split("_", 1)[1] if "_" in t[int(j)] else t[int(j)]
                for j in self.cols]

    # ---- the recorded run ----

    def _correr(self, force):
        etq = f"{self.name}_{self.grupo_nom}_ipc{self.ipc}_it{self.iters}_s{self.seed}"
        ruta = os.path.join(self.out, f"grabacion_{etq}.npz")
        if os.path.exists(ruta) and not force:
            z = np.load(ruta, allow_pickle=True)
            return (z["crudo"], z["visto"], z["grad"], torch.tensor(z["X_syn"]),
                    json.loads(str(z["meta"])))

        cfg = {"device": self.dev, "ipc": self.ipc, "dm_iters": self.iters,
               "dm_lr": self.lr, "dm_batch_real": 128,
               "dm_embedder_type": self.embedder, "dm_embedder_size": "base",
               "dm_embed_hidden": 256, "dm_embed_dim": max(4, min(48, self.ipc - 2)),
               "random_seed": self.seed, "init_seed": self.seed,
               "dm_onehot_mode": "ste"}
        t0 = time.time()
        _semilla(self.seed)
        with _RelojQuieto(), _Grabadora(self.cols.to(self.dev)) as rec:
            X_syn = synthesize("tame", self.data, cfg)[0].detach().cpu()
        crudo, visto, grad = rec.arrays()
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meta = dict(cfg, segundos=round(time.time() - t0, 1), dataset=self.name,
                    grupo=self.grupo_nom, llamadas=int(crudo.shape[0]),
                    git=os.popen(f"git -C {raiz} rev-parse --short HEAD").read().strip())
        np.savez_compressed(ruta, crudo=crudo, visto=visto, grad=grad,
                            X_syn=X_syn.numpy(), meta=json.dumps(meta))
        return crudo, visto, grad, X_syn, meta

    # ---- helpers ----

    def _score(self, V):
        """0 if the category is off, 1 if it is on. Works on any of the stacks."""
        return (V - self.z0.numpy()) / (self.z1.numpy() - self.z0.numpy())

    def _marco(self, filas, dec=3):
        f = pd.DataFrame(filas)
        return f.round(dec)

    # ---- the walkthrough ----

    def adelante(self, fila=0, it=0):
        """What comes out of ``ste``, and that it equals the hard projection.

        The identity ``X + (hard(X) - X) = hard(X)`` is trivial on paper and the
        point is that it holds on the numbers the run really used: the middle
        column is a constant as far as autograd is concerned
        """
        X, V = self.crudo[it, fila], self.visto[it, fila]
        return self._marco(dict(columna=self.nombres, syn_data=X,
                                constante=V - X, ve_el_embedder=V,
                                puntaje_antes=self._score(X),
                                puntaje_visto=self._score(V)))

    def atras(self, fila=0, it=0):
        """The gradient that came back, and where it landed.

        With a straight-through estimator the derivative through the projection
        is one, so the gradient measured on the tensor the embedder consumed is
        the gradient applied to the continuous variable, unchanged. The last
        column is the check: it is the same number
        """
        V, G = self.visto[it, fila], self.grad[it, fila]
        Xs = self.crudo[it + 1, fila] if it + 1 < len(self.crudo) else np.full_like(V, np.nan)
        X = self.crudo[it, fila]
        return self._marco(dict(columna=self.nombres, ve_el_embedder=V, gradiente=G,
                                syn_data_antes=X, syn_data_despues=Xs,
                                paso_observado=Xs - X), dec=4)

    def un_paso(self, fila=0, it=0):
        """One iteration end to end, as six rows of the same group.

        Everything comes from consecutive records of the real run, so the last
        row is not a simulation of the next step: it is the next step
        """
        X, V, G = self.crudo[it, fila], self.visto[it, fila], self.grad[it, fila]
        Xs = self.crudo[it + 1, fila] if it + 1 < len(self.crudo) else np.full_like(V, np.nan)
        Vs = self.visto[it + 1, fila] if it + 1 < len(self.visto) else np.full_like(V, np.nan)
        f = pd.DataFrame([X, V, G, Xs, Vs, self._score(X), self._score(Xs)],
                         columns=self.nombres)
        f.insert(0, "paso", [
            f"1. syn_data en it {it}",
            "2. lo que ve el embedder",
            "3. gradiente que vuelve",
            f"4. syn_data en it {it + 1}",
            "5. lo que vera el embedder",
            "6. puntaje antes",
            "7. puntaje despues"])
        return f.round(3)

    def identidad(self, n=200):
        """That the projection changed nothing on the rows that were legal.

        Run over the first ``n`` recorded iterations: for every entry whose score
        was already 0 or 1, what the embedder saw has to be the same number. This
        is the same property the evaluation-time experiment rests on, checked
        here against the loop's own traffic
        """
        X, V = self.crudo[:n], self.visto[:n]
        # Legal means the raw value IS one of the column's two values, not that it
        # rounds to one. A tolerance here would let a near miss count as legal and
        # then reappear as a near miss in the answer, which proves nothing.
        legal = (X == self.z0.numpy()) | (X == self.z1.numpy())
        if not legal.any():
            return dict(entradas_legales=0, identica=True, max_abs_dif=0.0)
        d = np.abs(V - X)[legal]
        return dict(entradas_legales=int(legal.sum()), identica=bool(d.max() == 0),
                    max_abs_dif=float(d.max()))

    def cambios(self):
        """When the argmax flipped, row by row.

        This is the mechanism the one-iteration tables cannot show. The variable
        drifts continuously and what the embedder sees is piecewise constant, so
        the interesting quantity is not how far the variable moved but how often
        that movement was enough to cross a score of 0.5 and change the winner
        """
        s = self._score(self.crudo)
        gana = s.argmax(-1)
        vivo = s.max(-1) >= 0.5
        etiqueta = np.where(vivo, gana, -1)
        filas = []
        for r in range(etiqueta.shape[1]):
            e = etiqueta[:, r]
            saltos = int((e[1:] != e[:-1]).sum())
            nom = lambda k: "apagado" if k < 0 else self.nombres[k]
            filas.append(dict(fila=r, empieza_en=nom(e[0]), termina_en=nom(e[-1]),
                              cambios=saltos,
                              ultimo_cambio=int(np.max(np.nonzero(e[1:] != e[:-1])[0]) + 1)
                              if saltos else -1))
        return pd.DataFrame(filas)

    def deriva(self):
        """How far the variable drifted while the embedder kept seeing legal rows.

        Two quantities per recorded iteration, averaged over rows and columns of
        the group: the distance from the raw variable to its nearest legal value,
        and the same distance for what the embedder was handed. The second is
        zero by construction, and printing it is how the claim gets checked
        rather than asserted
        """
        def dist(A):
            s = self._score(A)
            return np.abs(s - np.clip(s, 0, 1).round()).mean(axis=(1, 2))
        it = np.arange(len(self.crudo))
        return pd.DataFrame(dict(iteracion=it, syn_data=dist(self.crudo),
                                 ve_el_embedder=dist(self.visto))).round(5)

    def resumen(self):
        """The four numbers this whole trace exists to produce."""
        d = self.deriva()
        c = self.cambios()
        return pd.DataFrame([
            dict(medida="iteraciones grabadas", valor=len(self.crudo)),
            dict(medida="deriva de syn_data al inicio", valor=float(d.syn_data.iloc[0])),
            dict(medida="deriva de syn_data al final", valor=float(d.syn_data.iloc[-1])),
            dict(medida="deriva de lo que ve el embedder",
                 valor=float(d.ve_el_embedder.abs().max())),
            dict(medida="cambios de argmax, total", valor=int(c.cambios.sum())),
            dict(medida="filas que nunca cambiaron", valor=int((c.cambios == 0).sum())),
        ])

    # ---- saving ----

    def pasos(self, fila=0, it=0):
        return [("adelante", self.adelante(fila, it)),
                ("atras", self.atras(fila, it)),
                ("un_paso", self.un_paso(fila, it)),
                ("cambios", self.cambios()),
                ("deriva", self.deriva()),
                ("resumen", self.resumen())]

    def guardar(self, fila=0, it=0):
        """Every table to its own CSV, plus one markdown with all of them."""
        etq = f"{self.name}_{self.grupo_nom}_ipc{self.ipc}_s{self.seed}"
        dest = os.path.join(self.out, etq)
        os.makedirs(dest, exist_ok=True)
        md = [f"# El estimador straight-through, paso a paso ({self.name}, {self.grupo_nom})", "",
              f"- destilacion: ipc {self.ipc}, {self.iters} iteraciones, {self.embedder}, "
              f"semilla {self.seed}, {self.meta['segundos']}s, commit {self.meta['git']}",
              f"- la proyeccion sobre entradas ya legales: {self.identidad()}", ""]
        for nom, df in self.pasos(fila, it):
            df.to_csv(os.path.join(dest, f"{nom}.csv"), index=False)
            md += [f"## {nom}", "", df.head(40).to_markdown(index=False), ""]
        open(os.path.join(dest, "tablas.md"), "w").write("\n".join(md))
        return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--grupo", default="relationship")
    ap.add_argument("--fila", type=int, default=0)
    ap.add_argument("--it", type=int, default=None,
                    help="which iteration to walk through; the midpoint by default, "
                         "because iteration 0 is still the real rows it started from")
    ap.add_argument("--ipc", type=int, default=10)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=132)
    ap.add_argument("--embedder", default="ln_res_l")
    ap.add_argument("--out", default=SALIDA)
    ap.add_argument("--force", action="store_true", help="record again even if cached")
    a = ap.parse_args()

    tz = TrazaSTE(a.grupo, a.dataset, a.seed, a.ipc, a.iters, embedder=a.embedder,
                  out=a.out, force=a.force)
    it = a.it if a.it is not None else len(tz.crudo) // 2
    print("grupo:", tz.nombres)
    print("la proyeccion sobre entradas ya legales:", tz.identidad())
    for nom, df in tz.pasos(a.fila, it):
        print(f"\n### {nom}\n")
        print(df.head(30).to_markdown(index=False))
    print("\nguardado en", tz.guardar(a.fila, it))


if __name__ == "__main__":
    main()
