#!/bin/bash
# Runs any of the sweep scripts in parallel, one process per dataset.
#
# Distilling is only 14% of the time (2.5s per distillation); the other 86% are
# the classifier fits, which run on CPU over sets of 10 to 100 rows and do not
# make use of the 16 cores. Splitting by dataset does.
#
# Usage: bash scripts/run_sharded.sh <script.py> <out_dir> <workers> <dataset...> [-- extra args]
# E.g.:  bash scripts/run_sharded.sh run_snap_projection.py ~/tame_runs/snap_par 5 \
#         adult bank german -- --ipcs 10 50 100 --runs 3
set -u
SCRIPT=$1; OUT=$2; W=$3; shift 3
DS=(); EXTRA=()
for a in "$@"; do
  if [ "$a" = "--" ]; then EXTRA=("${@:$((${#DS[@]}+2))}"); break; fi
  DS+=("$a")
done
mkdir -p "$OUT"
# Each worker gets a share of the 16 cores; without this RF and XGBoost each ask
# for all of them and fight.
THREADS=$(( $(nproc) / W )); [ "$THREADS" -lt 1 ] && THREADS=1
echo "script=$SCRIPT  workers=$W  threads/worker=$THREADS  datasets=${#DS[@]}  extra=${EXTRA[*]:-}"
printf '%s\n' "${DS[@]}" | xargs -P "$W" -I{} sh -c "
  export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
  PYTHONPATH=/home/zxxz6/TAME /home/zxxz6/TAME/.venv/bin/python -u \
    /home/zxxz6/TAME/scripts/$SCRIPT --datasets {} --out $OUT/{} ${EXTRA[*]:-} \
    > $OUT/{}.log 2>&1 || echo 'FAILED {}' >> $OUT/errors.txt
"
echo "--- shards ---"
for d in "${DS[@]}"; do
  n=$( [ -f "$OUT/$d/results.csv" ] && tail -n +2 "$OUT/$d/results.csv" | wc -l || echo 0 )
  printf "  %-24s %5d rows %s\n" "$d" "$n" \
    "$(grep -q '^CSV ->' "$OUT/$d.log" 2>/dev/null && echo ok || echo CHECK)"
done
/home/zxxz6/TAME/.venv/bin/python - "$OUT" <<'PY'
import sys, glob, pandas as pd
out = sys.argv[1]
fs = sorted(glob.glob(f"{out}/*/results.csv"))
if not fs: sys.exit("no results")
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
df.to_csv(f"{out}/results.csv", index=False)
print(f"\nmerged {len(fs)} shards -> {out}/results.csv  ({len(df)} rows)")
PY
