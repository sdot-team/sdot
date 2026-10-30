#!/bin/bash
# Builds geogram + exploragram ( static, -march=native ) and the driver `geogram_ot` into $1 ( default
# ~/.cache/sdot/geogram-bench, where `bench_geogram.py` looks ). `build_baselines.py` calls it on each machine.
#
#   GEOGRAM_SRC  the geogram sources ( default ~/.cache/sdot/geogram-src, cloned when missing ); exploragram is cloned
#                into src/lib/exploragram ( a separate repository: https://github.com/BrunoLevy/exploragram )
#   LIBS         a prefix holding TBB ( -ltbb: libstdc++'s parallel algorithms ); empty: the system's
#   CXX          the compiler of the driver ( default g++ )
set -e
OUT=${1:-$HOME/.cache/sdot/geogram-bench}
SRC=${GEOGRAM_SRC:-$HOME/.cache/sdot/geogram-src}
HERE=$(cd "$(dirname "$0")" && pwd)
test -d "$SRC/.git" || git clone -q https://github.com/BrunoLevy/geogram.git "$SRC"
test -d "$SRC/src/lib/exploragram" || git clone -q https://github.com/BrunoLevy/exploragram.git "$SRC/src/lib/exploragram"
( cd "$SRC" && git submodule update --init --recursive -q ) || echo "== geogram: cannot update the submodules ( offline? ): building them as they are"
mkdir -p "$OUT/build" && cd "$OUT/build"
cmake "$SRC" -DVORPALINE_PLATFORM=Linux64-gcc -DCMAKE_BUILD_TYPE=Release -DGEOGRAM_WITH_GRAPHICS=OFF -DGEOGRAM_LIB_ONLY=ON \
      -DGEOGRAM_WITH_LUA=OFF -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_C_FLAGS="-march=native" > cmake.log
make -j"$(nproc)" geogram exploragram > make.log
LIBFLAGS=()
[ -n "$LIBS" ] && LIBFLAGS=( -I"$LIBS/include" -L"$LIBS/lib" -Wl,-rpath,"$LIBS/lib" )
# the same flags as the library ( geogram's predicates need -ffp-contract=off )
${CXX:-g++} -std=c++17 -O3 -march=native -ffp-contract=off -fopenmp -DNDEBUG -I"$SRC/src/lib" -I"$OUT/build/src/lib" \
    "$HERE/geogram_ot.cpp" -L"$OUT/build/lib" "${LIBFLAGS[@]}" -lexploragram -lgeogram -ltbb -lpthread -ldl -o "$OUT/geogram_ot"
printf '%s\n' "$(cd "$SRC" && git log -1 --format='geogram %h %ad' --date=short)" \
              "$(cd "$SRC/src/lib/exploragram" && git log -1 --format='exploragram %h %ad' --date=short)" > "$OUT/VERSION"
echo "built $OUT/geogram_ot"; cat "$OUT/VERSION"
