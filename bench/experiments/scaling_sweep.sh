#!/bin/bash
# Thread scaling of the diagram ( sdot vs CGAL ) and of the 2D linear solvers, on one node.
#   errand run --env n20-numpy --no-queue --pull runs/scaling-n20 bash sdot-ffi/bench/experiments/scaling_sweep.sh "1 2 4 8 16 18 36"
# The results land in runs/scaling-<host>/<bench>-T<threads>-<params>.
THREADS=${1:-"1 2 4 8 16"}
OUT=runs/scaling-${2:-n20}
B=sdot-ffi/bench
run() { python -m errand run -k bench --out "$OUT/$1-{params}" "$B/$2::*" "${@:3}"; }

echo "== $(hostname): $(grep -m1 'model name' /proc/cpuinfo); allowed CPUs: $(taskset -cp $$ | cut -d: -f2)"
lscpu | grep -E 'Socket|Core|Thread|NUMA|MHz' || true

python $B/build_baselines.py || echo "( the baselines could not be built )"

for T in $THREADS; do
    run diagram-T$T bench_diagram.py --case=uniform --dim=2,3 --threads=$T --kernels=compile --device=cpu
    run cgal-T$T bench_cgal.py --case=uniform --dim=2,3 --threads=$T
    run newton-T$T bench_newton.py --case=uniform --dim=2 --n=100000,1000000 --linear_solver=cholesky,mg --threads=$T \
        --kernels=compile --device=cpu --reps=1
done
