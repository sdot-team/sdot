#!/usr/bin/env bash
# THE NEWTON ABLATION ( COMPARISONS.md, B1-B6 ): the same cases and machine, the strategy changed one piece at a time,
# from the plain KMT damped Newton to the library's default. Run from ~/Projects; the log goes to stdout, the
# `result.yaml` of every run to runs/ ( `-o` ).
#
#   sdot-ffi/bench/ablation.sh [ ENV ] [ THREADS_FLAGS ] > ablation.log
#
# ENV defaults to lmo-numpy ( the CPU ). THREADS_FLAGS defaults to the campaign's `--threads=8 --pin=yes`.
set -uo pipefail

ENV="${1:-lmo-numpy}"
THREADS="${2:---threads=8 --pin=yes}"
PY="${PY:-$HOME/.mamba/envs/prj-numpy/bin/python}"
TARGET='sdot-ffi/bench/bench_newton.py::*'

run() { "$PY" -m errand run -o --no-queue --env "$ENV" "$TARGET" $THREADS --trace=yes "$@"; }

# the variants, each one adding a piece to the previous one
KMT="--step=trials --residual=lin --restart-factor=1e9 --aggregate=no"   # B1: KMT, every line search from t = 1
RESTART="--step=trials --residual=lin --aggregate=no"                    # + the line search restarts from the last step
LOG="--step=trials --residual=log --aggregate=no"                        # B2: + the log residual, switch to lin at 2
LIMITS="--step=limits --residual=log --aggregate=no"                     # B3: + the limits step ( 2D only )

# LEBESGUE, 2D ( n = 1e5, the files' clouds )
for v in "$KMT" "$RESTART" "$LOG" "$LIMITS" "${LIMITS/--aggregate=no/--aggregate=yes}"; do
    run --case=uniform,lines_voronoi,lines_equal --dim=2 --reps=3 $v
done
# LEBESGUE, 3D
for v in "$KMT" "$RESTART" "$LOG" "${LOG/--aggregate=no/--aggregate=yes}"; do
    run --case=uniform,planes_voronoi,planes_equal --dim=3 --reps=3 $v
done

# DENSITIES ( 2D, n = 1e5 for gauss4, 2e4 for the images ): every variant without, then with, the width continuation
for c in never always; do
    for v in "$KMT" "$RESTART" "$LOG" "$LIMITS" "${LIMITS/--aggregate=no/--aggregate=yes}"; do
        run --case=gauss4 --sigma=0.05,0.02,0.01 --continuation=$c --reps=1 $v
        run --case=image,image_hole --continuation=$c --reps=1 $v
    done
done

# B6, THE LINEAR SOLVERS on the default strategy
run --case=uniform,lines_voronoi,lines_equal --dim=2 --reps=3 --step=limits --linear-solver=cholesky,amg,mg
run --case=uniform,planes_voronoi,planes_equal --dim=3 --reps=3 --step=trials --linear-solver=cholesky,amg,mg
