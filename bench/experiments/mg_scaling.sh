#!/bin/bash
# The 2D Newton's linear part against the thread count: uniform 1e5 and 1e6, the lines, one linear solver ( `mg`, `auto`, ... ).
#   errand run --env n20-numpy --no-queue --pull runs/mg-scaling-auto bash sdot-ffi/bench/experiments/mg_scaling.sh "1 8 16 36" auto
# The results land in runs/mg-scaling-<solver>/T<threads>-<case>-<params>.
THREADS=${1:-"1 8 16"}
LIN=${2:-mg}
OUT=runs/mg-scaling-$LIN
B=sdot-ffi/bench
echo "== $(hostname): $(grep -m1 'model name' /proc/cpuinfo)"
for T in $THREADS; do
    for N in 100000 1000000; do
        echo "== T=$T uniform 2D n=$N"
        python -m errand run -k bench --out "$OUT/T$T-u$N-{params}" "$B/bench_newton.py::*" --case=uniform --dim=2 --n=$N \
            --linear_solver=$LIN --threads=$T --kernels=compile --device=cpu --reps=1 2>&1
    done
    echo "== T=$T lines"
    python -m errand run -k bench --out "$OUT/T$T-lines-{params}" "$B/bench_newton.py::*" --case=lines_voronoi \
        --linear_solver=$LIN --threads=$T --kernels=compile --device=cpu --reps=1 2>&1
done
