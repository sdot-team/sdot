#!/usr/bin/env bash
# CGAL AGAINST sdot ( COMPARISONS.md, A1 ), on lmo, the machine alone. Run from ~/Projects.
#
#   sdot-ffi/bench/cgal_vs_sdot.sh > cgal_vs_sdot.log
#
# Three outputs, each on both sides ( a fair comparison gives the same result to the user ):
#
#   CELLS        the measure of every cell. CGAL: triangulation + every cell rebuilt from its neighbours and clipped to the
#                box ( 2D: by hand; 3D `--cells`: with the old campaign's cut ). sdot: `bench_diagram`.
#   ADJACENCY    the neighbours of every cell. CGAL: triangulation + the tour of each vertex ( `power_2d --adjacency`,
#                `power_3d` without `--cells` ). sdot: `bench_diagram --output=adjacency` ( neighbours AND facet measures ),
#                and `--output=complex` ( the whole cell complex, glued ).
#   SEQUENCE     K diagrams on the same positions, the weights changing ( a Newton solve ). CGAL has nothing to reuse: its
#                cost per diagram is its CELLS chain on the final weights ( the files `bench_sequence --dump` writes ).
#                sdot: `bench_sequence`, the preparation once.
#
# CGAL is sequential: sdot is timed on 1 thread and on 8. CGAL's binaries are built in $CGAL_BENCH ( COMPARISONS.md ).
set -uo pipefail

CGAL_BENCH="${CGAL_BENCH:-/data/cgal-bench}"
CASES=/data/Projects-rsync/sdot-ffi/bench/cases
PY="${PY:-$HOME/.mamba/envs/prj-numpy/bin/python}"
THREADS=( "--threads=1" "--threads=8 --pin=yes" )

sd() { local f=$1; shift; "$PY" -m errand run -o --no-queue --env lmo-numpy "sdot-ffi/bench/$f::*" "$@"; }

echo "== sdot, SEQUENCE ( K = 10 ), the final weights dumped for CGAL"
for thr in "${THREADS[@]}"; do
    sd bench_sequence.py $thr --case=uniform --dim=2 --dump=$CGAL_BENCH/seq_uniform2d.txt
    sd bench_sequence.py $thr --case=uniform --dim=3 --n=1000000 --dump=$CGAL_BENCH/seq_uniform3d.txt
    for c in lines_voronoi lines_equal planes_voronoi planes_equal; do
        sd bench_sequence.py $thr --case=$c --dump=$CGAL_BENCH/seq_$c.txt
    done
done

echo "== CGAL ( $( ssh lmo "git -C $CGAL_BENCH/cgal log -1 --format='%h %ad' --date=short" ) )"
ssh lmo "bash -s" <<EOF
cd $CGAL_BENCH
for f in uniform2d:"1000000" lines_voronoi:"--load $CASES/lines5_n100000_s0.005_voronoi.txt" lines_equal:"--load $CASES/lines5_n100000_s0.005_equal.txt"; do
    echo "-- 2D \${f%%:*} CELLS";      ./power_2d \${f#*:} --reps 3
    echo "-- 2D \${f%%:*} ADJACENCY";  ./power_2d \${f#*:} --reps 3 --adjacency
    echo "-- 2D \${f%%:*} SEQUENCE";   ./power_2d --load seq_\${f%%:*}.txt --reps 3
done
for f in uniform3d_1e6:"1000000" uniform3d_1e5:"100000" planes_voronoi:"--load $CASES/planes4_n100000_s0.02_voronoi.txt" planes_equal:"--load $CASES/planes4_n100000_s0.02_equal.txt"; do
    echo "-- 3D \${f%%:*} CELLS + ADJACENCY";  ./power_3d \${f#*:} --reps 3 --cells
done
for f in uniform3d planes_voronoi planes_equal; do
    echo "-- 3D \$f SEQUENCE";        ./power_3d --load seq_\$f.txt --reps 3 --cells
done
EOF

echo "== sdot, CELLS and ADJACENCY"
for thr in "${THREADS[@]}"; do
    for out in measures adjacency; do
        sd bench_diagram.py $thr --output=$out --case=uniform,lines_voronoi,lines_equal --dim=2
        sd bench_diagram.py $thr --output=$out --case=uniform --dim=3 --n=1000000,100000
        sd bench_diagram.py $thr --output=$out --case=planes_voronoi,planes_equal
    done
done
echo "== sdot, the whole cell COMPLEX ( 8 threads, n = 1e5 )"
sd bench_diagram.py --threads=8 --pin=yes --output=complex --reps=1 --case=uniform --dim=2,3 --n=100000
