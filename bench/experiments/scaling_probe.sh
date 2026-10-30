#!/bin/bash
# Why the diagram does not scale linearly: the clock of the cores while it runs ( turbo, AVX-512 licence ), per SIMD level,
# and the hardware counters when `perf` is there; then the Cholesky / multigrid crossover of the 2D Newton on a clustered cloud.
#   errand run --env n20-numpy --no-queue --pull runs/scaling-probe bash sdot-ffi/bench/experiments/scaling_probe.sh
OUT=runs/scaling-probe
B=sdot-ffi/bench
run() { python -m errand run -k bench --out "$OUT/$1-{params}" "$B/$2::*" "${@:3}"; }

# the mean clock of the allowed CPUs, sampled every 0.2 s while a command runs ( in the background, to `$1.mhz` )
CPUS=$(taskset -cp $$ | cut -d: -f2 | tr -d ' ')
clock() {
    local log=$1; shift
    mkdir -p "$(dirname "$log")"
    ( while :; do awk -v cpus="$CPUS" 'BEGIN { n = split( cpus, r, "," ); for ( i = 1; i <= n; ++i ) { if ( split( r[ i ], ab, "-" ) == 2 ) for ( c = ab[ 1 ]; c <= ab[ 2 ]; ++c ) ok[ c ] = 1; else ok[ r[ i ] ] = 1 } }
                          /^processor/ { p = $3 } /^cpu MHz/ { if ( p in ok ) printf "%d ", $4 } END { print "" }' /proc/cpuinfo; sleep 0.2; done ) > "$log" &
    local pid=$!
    "$@"
    kill $pid
}

echo "perf: $(command -v perf || echo none)"
for TARGET in x86-64-v4 x86-64-v3; do
    for T in 1 8 16; do
        clock "$OUT/clock-$TARGET-T$T.mhz" run diagram-$TARGET-T$T bench_diagram.py --case=uniform --dim=3 --threads=$T \
            --kernels=compile --target=$TARGET --device=cpu --reps=5
    done
done

for T in 1 8 16; do
    run newton-T$T bench_newton.py --case=lines_voronoi,planes_voronoi --linear_solver=cholesky,mg --threads=$T \
        --kernels=compile --device=cpu --reps=1
    run newton-T$T bench_newton.py --case=uniform --dim=2 --n=30000,300000 --linear_solver=cholesky,mg --threads=$T \
        --kernels=compile --device=cpu --reps=1
done
