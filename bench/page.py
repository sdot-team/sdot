"""THE BENCHMARKS OF THE DOCUMENTATION ( `docs/guide/performance.md` ), on any machine errand reaches.

    cd ~/Projects                                          # the errand root
    python sdot-ffi/bench/page.py lmo-numpy lmo-jax        # the CPU suite in lmo-numpy, the GPU one in lmo-jax
    python sdot-ffi/bench/page.py jz-h100-jax              # both suites in one environment ( the CPU one with --device=cpu )
    python sdot-ffi/bench/page.py n22-jax --name n22-a6000 # one more machine
    python sdot-ffi/bench/page.py lmo-numpy --only diagram,cgal --quick   # a smoke run of two jobs
    python sdot-ffi/bench/page.py xyz-numpy --threads=1,8,-1             # 1, 8 threads and all the cores ( one more column each )
    python sdot-ffi/bench/page.py --collect lmo-numpy lmo-jax             # rewrite the results file only

    # a cluster ( Jean Zay ): page.py itself in one watched job, whose errand calls run in it directly
    errand run --env jz-h100-img --time 10:00:00 --pull runs/sdot-page --pull sdot-ffi/docs/guide/bench \
        python sdot-ffi/bench/page.py jz-h100-img --name jz-h100 --threads=1,8,-1 --no-queue

The CPU kernels are compiled for the machine ( `--kernels=compile` ), not served from the catalogue. An environment can
run the `cpu` jobs ( any: they get `--device=cpu`, so an environment with a card computes them on the CPU ) and, when its
driver runs on the card ( jax with `cuda` ), the `gpu` ones; each kind runs in the FIRST environment listed that can run
it, so one environment with a card runs everything. `--kind` keeps one kind only. Every job is one `errand run` of a bench of this directory, with
`-x` ( the machine to itself; `--no-queue` to skip the wait ), its runs recorded under
`runs/sdot-page/<env>/<stamp>/<job>/`.

The results land in `docs/guide/bench/<name>.json` ( `<name>`: `--name`, by default the prefix of the first environment,
`lmo` for `lmo-numpy` ), which the page reads. Rows of an environment replace the rows of the same environment already in
the file, so the CPU and the GPU suites of one machine can run at different times; within an environment, each job comes
from the last batch that ran it, so `--only` reruns a few jobs without losing the others.

`--quick`: small sizes and one repetition, to check that everything runs ( the file is written to `<name>-quick.json`,
which the page ignores ). `--full`: also the long rows ( 6D, 1e7 on the card ).

The CGAL and geogram jobs run programs built on the machine: before them, `build_baselines.py` builds what is missing or
out of date ( into `~/.cache/sdot/`, the libraries in a micromamba environment of their own; `--no-build` skips it ). A
program that cannot be built makes its rows failed, and the page shows a dash.

= On a new machine

1. A `Machine( ... )` line in `errand-envs.py` ( its prefix, say `xyz`, names the environments `xyz-numpy`, `xyz-jax` ),
   then `errand setup --env xyz-numpy` ( or `xyz-jax` alone for a card: it runs both suites ).
2. `python sdot-ffi/bench/page.py xyz-numpy --quick`: every job, small, in a few minutes ( `docs/guide/bench/xyz-quick.json` ).
3. `python sdot-ffi/bench/page.py xyz-numpy xyz-jax`: the page's numbers, `docs/guide/bench/xyz.json`. The page shows every
   machine of `docs/guide/bench/` ( `bench.data.js` ), side by side.

The clouds of the lines / planes cases are files ( `bench/cases/`, not in git, copied by errand's rsync with the rest of the
tree ): `cases.py` says where else they are looked for.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path( __file__ ).resolve().parent
DOCS = HERE.parent / "docs" / "guide" / "bench"

BASELINES = { "cgal", "geogram" }      # the benches whose programs `build_baselines.py` builds on the machine
FILES = { "diagram": "bench_diagram.py", "newton": "bench_newton.py", "sequence": "bench_sequence.py", "cgal": "bench_cgal.py", "geogram": "bench_geogram.py" }

# the clouds of `bench_diagram` / `bench_cgal` / `bench_geogram` ( a list of runs: one errand call each, a comma is a matrix )
CLOUDS = [
    "--case=uniform --dim=2 --wscale=0,0.3",
    "--case=uniform --dim=3",
    "--case=lines_voronoi,lines_equal,planes_voronoi,planes_equal",
]
DENSITIES = [
    "--case=image --reps=1",
    "--case=gauss4 --sigma=0.02 --reps=1",
]

# job -> ( kind, bench, [ runs ], level ). level: `default` or `full`. `{T}`: the thread counts of `--threads` ( default 1 and
# the physical cores, -1 ); the others run on all the cores.
JOBS = {
    # ---------------------------------------------------------------------------------------------------- the CPU
    "diagram":    ( "cpu", "diagram",  [ c + " --threads={T}" for c in CLOUDS ], "default" ),
    "cgal":       ( "cpu", "cgal",     [ c + " --output=measures --threads={T}" for c in CLOUDS ], "default" ),
    "geogram":    ( "cpu", "geogram",  [ c + " --threads={T}" for c in CLOUDS ], "default" ),
    # the exact dual: sdot's `dual()`, CGAL's triangulation ( geogram's is the `triangulation` half of the job `geogram` )
    "dual":       ( "cpu", "diagram",  [ c + " --threads={T} --output=dual" for c in CLOUDS ], "default" ),
    "cgal_dual":  ( "cpu", "cgal",     [ "--case=uniform --dim=2 --wscale=0,0.3 --output=triangulation",
                                         "--case=lines_voronoi,lines_equal --output=triangulation",
                                         "--case=uniform --dim=3 --output=triangulation --threads={T}",
                                         "--case=planes_voronoi,planes_equal --output=triangulation --threads={T}" ], "default" ),
    "geogram_newton": ( "cpu", "geogram", [ "--case=uniform --dim=2,3 --level=newton --threads={T}",
                                            "--case=lines_voronoi,planes_voronoi --level=newton --threads={T}" ], "default" ),
    "sequence":   ( "cpu", "sequence", [ "--case=uniform --dim=2,3 --threads={T}" ], "default" ),
    "newton":     ( "cpu", "newton",   [ "--case=uniform --dim=2,3 --threads={T} --step=auto",       # ( every count: the comparison with geogram )
                                         "--case=lines_voronoi,planes_voronoi --threads={T} --step=auto",
                                         *[ d + " --threads=-1 --step=auto" for d in DENSITIES ] ], "default" ),
    "dims":       ( "cpu", "newton",   [ "--case=uniform --dim=2,3,4,5 --n=10000 --threads=-1 --reps=1" ], "default" ),
    "dims6":      ( "cpu", "newton",   [ "--case=uniform --dim=6 --n=10000 --threads=-1 --reps=1" ], "full" ),
    # ---------------------------------------------------------------------------------------------------- the card
    "gpu_diagram": ( "gpu", "diagram", [ "--case=uniform --dim=2 --wscale=0,0.3 --kernel=float,double",
                                         "--case=uniform --dim=3 --kernel=float,double",
                                         "--case=lines_voronoi,lines_equal,planes_voronoi,planes_equal --kernel=float,double" ], "default" ),
    "gpu_dual":    ( "gpu", "diagram", [ "--case=uniform --dim=2 --wscale=0,0.3 --kernel=double --output=dual",
                                         "--case=uniform --dim=3 --kernel=double --output=dual",
                                         "--case=lines_voronoi,lines_equal,planes_voronoi,planes_equal --kernel=double --output=dual" ], "default" ),
    "gpu_newton":  ( "gpu", "newton",  [ "--case=uniform --dim=2,3 --n=100000,1000000 --kernel=mixed --jit=yes --step=auto",
                                         "--case=lines_voronoi,planes_voronoi --kernel=mixed --jit=yes --step=auto",
                                         *[ d + " --kernel=mixed --jit=yes --step=auto" for d in DENSITIES ] ], "default" ),
    "gpu_n1e7":    ( "gpu", "newton",  [ "--case=uniform --dim=2,3 --n=10000000 --kernel=mixed --jit=no --reps=1 --step=auto" ], "full" ),
}

# `--quick`: the flags each bench gets instead of its own
QUICK = {
    "diagram":  "--n=20000 --reps=1",
    "cgal":     "--n=20000 --reps=1",
    "geogram":  "--n=20000 --reps=1",
    "sequence": "--n=20000 --reps=1",
    "newton":   "--n=3000 --reps=1 --image-size=64",
}

# result keys worth keeping ( the others are diagnostics of the benches )
KEEP = { "n", "dim", "threads", "kernel", "output", "status", "converged", "iterations", "diagrams", "ns_per_seed",
         "ns_per_seed_kernel", "ns_per_seed_wall", "t_total", "t_total_wall", "t_jit_wall", "t_eager_wall", "t_diag", "t_lin",
         "t_tree", "t_prepare", "t_first", "t_next", "ns_per_seed_next", "ns_per_seed_mean", "sequence", "acc_median",
         "acc_max", "nb_continuation_steps", "residual_rel", "step", "linear_solver", "program", "cgal_commit", "t_bsp_build",
         "geogram", "t_triangulation", "t_rvd", "t_linear", "max_diff_sdot", "residual_sdot", "grid", "residual", "level", "triangulations",
         "mean_neighbours", "simplices" }


def errand_root():
    d = Path.cwd().resolve()
    for p in ( d, *d.parents ):
        if any( p.glob( "errand-*.py" ) ):
            return p
    sys.exit( "page.py: run it from inside an errand project ( no errand-*.py above the current directory )" )


def env_kinds( root, name ):
    """the kinds of jobs the environment can run: `cpu` always ( `--device=cpu` ), `gpu` too when its driver runs on the card"""
    from errand import config
    config.load( root )
    if name not in config.envs:
        sys.exit( f"page.py: no errand environment { name !r } ( { ', '.join( sorted( config.envs ) ) } )" )
    tags = config.envs[ name ].tags
    return ( "cpu", "gpu" ) if tags.get( "cuda" ) and tags.get( "driver" ) in ( "jax", "torch", "cupy" ) else ( "cpu", )


def with_quick( run, bench, quick ):
    """the flags of a run, those of `QUICK[ bench ]` replacing its own under `--quick`"""
    flags = run.split()
    if quick:
        extra = QUICK.get( bench, "" ).split()
        keys = { f.split( "=" )[ 0 ] for f in extra }
        flags = [ f for f in flags if f.split( "=" )[ 0 ] not in keys ] + extra
    return flags


# flags read once per PROCESS ( the thread pool, the served kernels ): errand runs the combinations of a matrix one after the
# other in ONE process, so a comma in them is split here into separate calls
PROCESS_WIDE = ( "--threads", "--pin", "--kernels", "--target" )


def split_process_wide( flags ):
    """the calls a run becomes: one per value of each process-wide flag"""
    calls = [ [ ] ]
    for f in flags:
        k, eq, v = f.partition( "=" )
        if eq and k in PROCESS_WIDE and "," in v:
            calls = [ c + [ f"{ k }={ x }" ] for c in calls for x in v.split( "," ) ]
        else:
            calls = [ c + [ f ] for c in calls ]
    return calls


def run_jobs( root, env, kinds, jobs, stamp, args ):
    rel = os.path.relpath( HERE, root )
    with_threads = lambda r: r.replace( "{T}", args.threads )
    queue = [ "--no-queue" ] if args.no_queue else [ "-x" ]
    if not args.no_build and any( JOBS[ j ][ 0 ] in kinds and JOBS[ j ][ 1 ] in BASELINES for j in jobs ):
        cmd = [ sys.executable, "-m", "errand", "run", "--env", env, *queue, f"{ rel }/build_baselines.py" ]
        print( f"\n== { env }: the programs of CGAL and geogram ( { ' '.join( cmd[ 2: ] ) } )", flush = True )
        if subprocess.call( cmd, cwd = root ):
            print( "   ( they could not be built: their jobs will fail )", flush = True )
    for job in jobs:
        jkind, bench, runs, _ = JOBS[ job ]
        if jkind not in kinds:
            continue
        on_cpu = [ "--kernels=compile", "--device=cpu" ] if jkind == "cpu" and bench not in BASELINES else [ ]
        flags = lambda r: [ *with_quick( r, bench, args.quick ), *on_cpu ]
        for call in ( c for r in runs for c in split_process_wide( flags( with_threads( r ) ) ) ):
            out = f"runs/sdot-page/{ env }/{ stamp }/{ job }/{{name}}-{{params}}"
            cmd = [ sys.executable, "-m", "errand", "run", "--env", env, "-k", "bench", "--out", out,
                    *queue, f"{ rel }/{ FILES[ bench ] }::*", *call ]
            print( f"\n== { env } { job }: { ' '.join( cmd[ 2: ] ) }", flush = True )
            code = subprocess.call( cmd, cwd = root )
            if code:
                print( f"   ( exit { code }: the failed runs are recorded as such )", flush = True )


def collect( root, envs, name, quick ):
    from errand import yamlish
    path = DOCS / f"{ name }{ '-quick' if quick else '' }.json"
    data = json.loads( path.read_text() ) if path.exists() else { "name": name, "rows": [ ], "envs": { } }
    for env in envs:
        base = root / "runs" / "sdot-page" / env
        stamps = sorted( d.name for d in base.iterdir() if d.is_dir() ) if base.exists() else [ ]
        if not stamps:
            print( f"page.py: nothing recorded for { env } under { base }" )
            continue
        # each job from the LAST batch that ran it: `--only` reruns some jobs without losing the others
        latest = { j.name: s for s in stamps for j in ( base / s ).iterdir() if j.is_dir() and j.name in JOBS }   # ( not the jobs gone )
        stamp = stamps[ -1 ]
        rows = [ ]
        data[ "envs" ].pop( env, None )
        for res in sorted( r for j, s in latest.items() for r in ( base / s / j ).glob( "*/result.yaml" ) ):
            if res.parent.is_symlink():                      # errand's `latest`, a second path to a run
                continue
            rec = yamlish.load( res.read_text() )
            r = rec.get( "results" ) or { }
            job = res.parent.parent.name
            kind, bench = JOBS[ job ][ :2 ]
            hw = { k[ 3: ]: v for k, v in r.items() if k.startswith( "hw_" ) and not ( bench in BASELINES and k == "hw_kernels" ) }
            if hw:                                           # ( the kernels of a baseline's run say nothing: it runs no loom kernel )
                data[ "envs" ].setdefault( env, { } ).setdefault( "kinds", { } ).setdefault( kind, { } ).setdefault( "hw", { } ).update( hw )
            rows.append( dict( env = env, job = job, kind = kind, bench = rec.get( "name" ), status = rec.get( "status" ),
                               params = rec.get( "params" ) or { }, results = { k: v for k, v in r.items() if k in KEEP } ) )
        data[ "envs" ].setdefault( env, { } ).update( stamp = stamp, stamps = latest,
                                                      date = stamp.split( "_" )[ 0 ], commit = git_commit() )
        data[ "rows" ] = [ x for x in data[ "rows" ] if x[ "env" ] != env ] + rows
        print( f"page.py: { len( rows ) } runs of { env } ( { stamp } )" )
    data[ "updated" ] = datetime.date.today().isoformat()
    path.parent.mkdir( parents = True, exist_ok = True )
    path.write_text( json.dumps( data, indent = 1 ) + "\n" )
    print( f"page.py: wrote { path }" )


def git_commit():
    r = subprocess.run( [ "git", "-C", str( HERE ), "describe", "--always", "--dirty" ], capture_output = True, text = True )
    return r.stdout.strip() or None


def main():
    ap = argparse.ArgumentParser( description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter )
    ap.add_argument( "envs", nargs = "+", help = "errand environments ( each kind of job runs in the first one that can run it )" )
    ap.add_argument( "--name", default = None, help = "the machine's name on the page ( default: the prefix of the first environment )" )
    ap.add_argument( "--kind", choices = [ "cpu", "gpu" ], default = None, help = "run the jobs of this kind only" )
    ap.add_argument( "--only", default = "", help = f"comma-separated jobs ( { ', '.join( JOBS ) } )" )
    ap.add_argument( "--full", action = "store_true", help = "also the long jobs ( " + ", ".join( j for j, v in JOBS.items() if v[ 3 ] == "full" ) + " )" )
    ap.add_argument( "--quick", action = "store_true", help = "small sizes, one repetition: a smoke run ( written to <name>-quick.json )" )
    ap.add_argument( "--threads", default = "1,-1", help = "the thread counts of the CPU sweeps ( -1: the physical cores ), e.g. 1,8,-1" )
    ap.add_argument( "--no-build", action = "store_true", help = "do not build CGAL's and geogram's programs first ( build_baselines.py )" )
    ap.add_argument( "--no-queue", action = "store_true", help = "do not wait for the machine to be free ( timings may suffer )" )
    ap.add_argument( "--collect", action = "store_true", help = "do not run: rewrite the results file from the last recorded runs" )
    args = ap.parse_args()

    root = errand_root()
    name = args.name or args.envs[ 0 ].split( "-" )[ 0 ]
    jobs = [ j for j in ( args.only.split( "," ) if args.only else JOBS ) if j ]
    if bad := [ j for j in jobs if j not in JOBS ]:
        sys.exit( f"page.py: unknown jobs { bad } ( { ', '.join( JOBS ) } )" )
    if not args.only:
        jobs = [ j for j in jobs if JOBS[ j ][ 3 ] == "default" or args.full ]

    if not args.collect:
        stamp = datetime.datetime.now().strftime( "%Y-%m-%d_%Hh%Mm%S" )
        todo = [ args.kind ] if args.kind else [ "cpu", "gpu" ]
        for env in args.envs:
            kinds = [ k for k in env_kinds( root, env ) if k in todo ]
            todo = [ k for k in todo if k not in kinds ]
            if kinds:
                run_jobs( root, env, kinds, jobs, stamp, args )
            else:
                print( f"page.py: nothing left for { env } ( its kinds run in an environment listed before )", flush = True )
    collect( root, args.envs, name, args.quick )


if __name__ == "__main__":
    main()
