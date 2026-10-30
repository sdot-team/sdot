#!/usr/bin/env bash
# The CGAL programs of `bench_cgal.py` ( `power_2d`, `power_3d` ), compiled against CGAL's git sources, into $1
# ( default ~/.cache/sdot/cgal-bench, where `bench_cgal.py` looks ). `build_baselines.py` calls it on each machine.
#
# From git and not from a package: CGAL moves often, and a baseline that compares against a version from two years ago
# does not tell what we want to know. CGAL 6 is a HEADER library: nothing to build in the clone, only a `-I` per package.
#
#   CGAL_DIR   where the clone lands ( default ~/.cache/sdot/cgal ); CGAL_REF its branch ( default main )
#   LIBS       the prefix of gmp / mpfr / boost / TBB ( default $CONDA_PREFIX: a micromamba env holding them )
#   CXX        the compiler ( default g++ )
#
# `power_3d --cells` uses the old campaign's cell ( `nsdot-src/`, a copy of `2d_des_familles/src/{geometry,util}` ).
set -euo pipefail

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUT="${1:-$HOME/.cache/sdot/cgal-bench}"
CGAL_DIR="${CGAL_DIR:-$HOME/.cache/sdot/cgal}"
CGAL_REF="${CGAL_REF:-main}"
LIBS="${LIBS:-${CONDA_PREFIX:-}}"
if [ -z "$LIBS" ]; then
    echo "!! no LIBS nor active micromamba env: gmp / mpfr / boost not found" >&2
    exit 1
fi

if [ -d "$CGAL_DIR/.git" ]; then
    echo "== CGAL: updating $CGAL_DIR ($CGAL_REF)"
    # offline, or a clone we may not write ( an image's ): built as it is
    if git -C "$CGAL_DIR" fetch -q --depth 1 origin "$CGAL_REF" 2> /dev/null; then
        git -C "$CGAL_DIR" checkout -q FETCH_HEAD
    else
        echo "   cannot fetch: building the clone as it is"
    fi
else
    echo "== CGAL: cloning into $CGAL_DIR ($CGAL_REF)"
    mkdir -p "$( dirname "$CGAL_DIR" )"
    git clone -q --depth 1 --branch "$CGAL_REF" https://github.com/CGAL/cgal.git "$CGAL_DIR"
fi
echo "   $( git -C "$CGAL_DIR" log -1 --format='%h %ad %s' --date=short )"

# a git checkout of CGAL is SPLIT into packages ( Kernel_23/include, Triangulation_2/include, ... ), a release glues them
# under one `include/`
incs=()
if [ -d "$CGAL_DIR/Installation/include" ]; then
    for d in "$CGAL_DIR"/*/include; do
        [ -d "$d" ] && incs+=( -I "$d" )
    done
else
    incs+=( -I "$CGAL_DIR/include" )
fi

mkdir -p "$OUT"
for b in power_2d power_3d; do
    echo "== compiling $b"
    # TBB: CGAL's parallel 3D triangulation; OpenMP: the cells over the threads ( `--threads` )
    ${CXX:-g++} -std=c++20 -O3 -march=native -fopenmp -DNDEBUG -DCGAL_NDEBUG -DCGAL_LINKED_WITH_TBB "${incs[@]}" -I "$here/nsdot-src" -I "$LIBS/include" \
        "$here/$b.cpp" -L "$LIBS/lib" -Wl,-rpath,"$LIBS/lib" -lgmp -lmpfr -ltbb -ltbbmalloc -o "$OUT/$b"
done
git -C "$CGAL_DIR" log -1 --format='cgal %h %ad' --date=short > "$OUT/VERSION"
echo "built $OUT/power_2d $OUT/power_3d"; cat "$OUT/VERSION"
