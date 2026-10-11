"""How much each one-hot mode gains over plain TAME, from the ablation's results.

`run_onehot_modes.py` writes one row per distillation and classifier. This turns
that into the four readings the `ste` experiment was judged on in the notebook,
now for every mode at once, so the two experiments can be read side by side:

  vs_tame          paired delta against `none`, the published loop: same dataset,
                   embedder, IPC, seed and classifier. Per block and classifier,
                   with n, mean, median, wins and a Wilcoxon test. Notebook:
                   `tabla_bucle`.
  robustez         that delta split by embedder and IPC, trees only. Notebook:
                   `tabla_robustez`.
  vs_filas_reales  accuracy minus that of the run's own initialisation, which is
                   real rows sampled at random, so "did distilling beat sampling".
                   Trees only, with the count of runs that beat it. Notebook:
                   `contra_filas_reales`.
  dano_vs_arreglo  per dataset, how far plain TAME falls below random rows (the
                   damage) against how much each mode adds (the repair), and the
                   correlation between the two. Notebook: `efecto_vs_dano`.

Definitions are copied from the notebook on purpose, thresholds included (a win
over `none` is any positive delta; a win over random rows needs more than 0.002),
so a number here means the same thing as the one in the `ste` section.

An arm is a mode and an evaluation view. `none` raw is the baseline; `none`
projected at evaluation time is the post-hoc fix of the first experiment, so it
is reported as one more arm, and in-loop against post-hoc lands in one table.

It needs every dataset at once, since a correlation across datasets cannot be
computed shard by shard, so it runs on the merged CSV. `run_sharded.sh` calls it
after merging when given ``POST=analyze_onehot_modes.py``; by hand:

    .venv/bin/python scripts/analyze_onehot_modes.py ~/tame_runs/onehot_modes/results.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd
from scipy import stats

ARB = ["rf", "xgboost"]
CLFS = ["rf", "xgboost", "mlp"]
BLOQUES = ["binario-banco", "binario-sin-onehot", "multi-onehot", "multi-sin-onehot"]
LLAVE = ["bloque", "dataset", "embedder", "ipc", "classifier", "run"]


def _col_init(metric):
    """The initialisation column that pairs with a metric: acc_init for both the
    returned set and the last iterate."""
    return metric.split("_")[0] + "_init"


def brazos_largos(df, metric):
    """One column per arm, one row per paired run, plus the baseline and init.

    The arm label is the mode, with ``+proy`` when it was evaluated projected.
    `none` raw is the baseline and is kept as its own column, not as an arm
    """
    d = df.copy()
    d["arm"] = d.brazo + np.where(d.eval_proj == "proyectado", "+proy", "")
    p = d.pivot_table(index=LLAVE, columns="arm", values=metric)
    # The initialisation is real rows and does not depend on the arm: every arm
    # starts from the same init_seed. Taken from the baseline's own rows.
    base = d[(d.brazo == "none") & (d.eval_proj == "crudo")].set_index(LLAVE)
    p["_init"] = base[_col_init(metric)]
    return p


DISCRETOS = ("ste", "soft", "project")


def consistencia(df, metric="acc_bestloss"):
    """That the discrete modes are unchanged by projecting them again.

    They return legal rows, so the projected view must be the same number to the
    last digit. The ablation asserts it on the tensors; this checks it on the
    accuracies, and is the reason their projected copies are then dropped from
    every table: they are a check, not arms
    """
    d = df[df["mode"].isin(DISCRETOS)]
    if d.eval_proj.nunique() < 2:
        return np.nan
    p = d.pivot_table(index=LLAVE + ["brazo"], columns="eval_proj", values=metric)
    return float((p["crudo"] - p["proyectado"]).abs().max())


def pares(df, metric="acc_bestloss"):
    """Long table: one row per paired run and arm, with every quantity derived."""
    p = brazos_largos(df, metric)
    filas = []
    sobran = {f"{m}+proy" for m in DISCRETOS}
    for arm in [c for c in p.columns if c not in ("none", "_init") and c not in sobran]:
        q = p[[arm, "none", "_init"]].dropna(subset=[arm, "none"]).copy()
        q.columns = ["acc", "acc_none", "acc_init"]
        q["arm"] = arm
        filas.append(q.reset_index())
    out = pd.concat(filas, ignore_index=True)
    out["delta"] = out.acc - out.acc_none                 # what the arm adds over plain TAME
    out["vs_reales"] = out.acc - out.acc_init             # what it adds over sampling real rows
    out["dano"] = out.acc_none - out.acc_init             # what plain TAME loses against sampling
    return out


def vs_tame(pr):
    """Paired delta against `none`, per arm, block and classifier. The notebook's
    `tabla_bucle`, for every arm."""
    filas = []
    for arm in sorted(pr.arm.unique()):
        for b in BLOQUES:
            for clf in CLFS:
                v = pr[(pr.arm == arm) & (pr.bloque == b) & (pr.classifier == clf)].delta.values
                if not len(v):
                    continue
                filas.append(dict(
                    arm=arm, bloque=b, clasificador=clf, n=len(v), media=v.mean(),
                    mediana=np.median(v), ic95=1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                    gana=int((v > 0).sum()),
                    wilcoxon=stats.wilcoxon(v)[1] if np.ptp(v) > 0 else np.nan))
    return pd.DataFrame(filas)


def robustez(pr):
    """The same delta by embedder and IPC, trees only. The notebook's
    `tabla_robustez`, for every arm."""
    t = pr[pr.classifier.isin(ARB)]
    return t.pivot_table(index=["arm", "bloque", "ipc"], columns="embedder",
                         values="delta", aggfunc="mean").reset_index()


def vs_filas_reales(df, pr, metric="acc_bestloss"):
    """Gain over the run's own initialisation, trees only, `none` included.

    `none` has no paired delta against itself, so its gain is read from the raw
    rows; the arms come from the paired table. Same threshold as the notebook:
    beating random rows takes more than 0.002
    """
    base = df[(df.brazo == "none") & (df.eval_proj == "crudo") & df.classifier.isin(ARB)]
    g_none = (base[metric] - base[_col_init(metric)]).rename("vs_reales")
    filas = []
    for arm, s in [("none", base.assign(vs_reales=g_none))] + \
                  [(a, pr[(pr.arm == a) & pr.classifier.isin(ARB)]) for a in sorted(pr.arm.unique())]:
        for b in BLOQUES:
            v = s[s.bloque == b].vs_reales.values
            if not len(v):
                continue
            filas.append(dict(arm=arm, bloque=b, n=len(v), media=v.mean(),
                              ic95=1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                              le_gana=int((v > 0.002).sum()),
                              le_gana_txt=f"{int((v > 0.002).sum())}/{len(v)}"))
    return pd.DataFrame(filas)


def dano_vs_arreglo(df, pr):
    """Per dataset, damage against repair, and their correlation per arm.

    Damage is how far plain TAME falls below random rows on that dataset; repair
    is what the arm adds over plain TAME. If an arm repairs a specific defect
    rather than raising the ceiling, repair should track damage and the datasets
    with no damage should sit at the origin. The dull alternative, that the
    effect just follows how many one-hot columns there are, is checked too, with
    the share of dummy columns
    """
    t = pr[pr.classifier.isin(ARB)]
    por_ds = (t.groupby(["arm", "bloque", "dataset"])[["dano", "delta"]].mean().reset_index())
    frac = (df.drop_duplicates("dataset").set_index("dataset")
            .eval("n_dummy / input_dim").rename("frac_dummy"))
    por_ds = por_ds.join(frac, on="dataset")
    corr = []
    for arm, s in por_ds.groupby("arm"):
        con = s[s.frac_dummy > 0]
        r = lambda x, y: (stats.pearsonr(x, y)[0] if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
                          else np.nan)
        corr.append(dict(arm=arm, n_datasets=len(s),
                         r_dano=r(s.dano, s.delta),
                         r_dano_solo_onehot=r(con.dano, con.delta),
                         r_frac_dummy_solo_onehot=r(con.frac_dummy, con.delta)))
    return por_ds, pd.DataFrame(corr)


def caidas(df):
    """Share of distillations whose returned set was the initialisation.

    Then the arm returned real rows and its accuracy says nothing about the mode.
    For `penalty` this was a bias of the selection and is fixed; for the others it
    is the selection working as designed when no iterate beat the starting point,
    which happened on `splice` in 9% of `soft` runs during tuning. Either way it
    has to be visible next to the deltas, not discovered after
    """
    d = df[df.eval_proj == "crudo"].drop_duplicates(["dataset", "embedder", "ipc", "brazo", "run"])
    return (d.pivot_table(index="brazo", columns="bloque", values="best_en_init", aggfunc="mean")
            .reindex(columns=[b for b in BLOQUES if b in d.bloque.unique()]).reset_index())


def analizar(csv, out=None, metric="acc_bestloss"):
    df = pd.read_csv(csv)
    out = out or os.path.join(os.path.dirname(os.path.abspath(csv)), "analisis")
    os.makedirs(out, exist_ok=True)
    pr = pares(df, metric)
    tablas = {"pares": pr, "vs_tame": vs_tame(pr), "robustez": robustez(pr),
              "vs_filas_reales": vs_filas_reales(df, pr, metric)}
    tablas["dano_vs_arreglo"], tablas["correlacion"] = dano_vs_arreglo(df, pr)
    tablas["caidas"] = caidas(df)
    for nom, t in tablas.items():
        t.to_csv(os.path.join(out, f"{nom}.csv"), index=False)

    pd.set_option("display.width", 240)
    print(f"{len(df)} filas, {df.dataset.nunique()} datasets, metrica {metric} -> {out}")
    c = consistencia(df, metric)
    print(f"consistencia: ste/soft/project proyectados contra crudos, max|dif| = {c}"
          + ("  OK, se omiten de las tablas" if c == 0 else "  ATENCION: deberian ser identicos"))
    v = tablas["vs_tame"]
    print("\n=== cuanto gana cada brazo sobre TAME normal, arboles (media pareada) ===")
    print(v[v.clasificador.isin(ARB)].pivot_table(index="arm", columns="bloque", values="media")
          .reindex(columns=BLOQUES).round(4).to_string())
    print("\n=== lo mismo, mlp ===")
    print(v[v.clasificador == "mlp"].pivot_table(index="arm", columns="bloque", values="media")
          .reindex(columns=BLOQUES).round(4).to_string())
    f = tablas["vs_filas_reales"]
    print("\n=== sobre filas reales al azar, arboles (media | corridas que le ganan) ===")
    print(f.pivot_table(index="arm", columns="bloque", values="media").reindex(columns=BLOQUES)
          .round(4).to_string())
    print(f.pivot_table(index="arm", columns="bloque", values="le_gana_txt", aggfunc="first")
          .reindex(columns=BLOQUES).to_string())
    print("\n=== dano contra arreglo: correlacion por brazo, arboles ===")
    print(tablas["correlacion"].round(3).to_string(index=False))
    print("\n=== fraccion de destilaciones que devolvieron la inicializacion ===")
    print(tablas["caidas"].round(3).to_string(index=False))
    return tablas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="the merged results.csv of run_onehot_modes.py")
    ap.add_argument("--out", default=None, help="defaults to <csv dir>/analisis")
    ap.add_argument("--metric", default="acc_bestloss",
                    choices=["acc_bestloss", "acc_final"],
                    help="the returned set, or the last iterate")
    a = ap.parse_args()
    analizar(a.csv, a.out, a.metric)
