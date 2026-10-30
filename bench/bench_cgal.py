"""CGAL ON THE SAME CLOUDS: the programs of `cgal/` ( `Regular_triangulation_2/3`, EPICK, bulk insertion ) run on the
cloud `bench_diagram` builds for the same parameters, so that both sides compute the same thing on the same points.

    errand run --env lmo-numpy 'sdot-ffi/bench/bench_cgal.py::*' --case=uniform --dim=2 --wscale=0,0.3 --output=measures,adjacency

`--output`:
  measures    every cell's measure. 2D: triangulation + each cell rebuilt from its neighbours and clipped to the unit square
              ( `power_2d` ); 3D: `power_3d --cells` ( the triangulation, then the old campaign's cut ).
  adjacency   the neighbours of every cell: triangulation + the tour of each vertex.
  triangulation  the regular triangulation alone: the exact dual ( what sdot's `dual()` gives, there restricted to the box ).
              3D with `--threads` > 1: CGAL's parallel triangulation ( `Parallel_tag`, TBB ); 2D: sequential only.

`--threads` > 1 with `measures`: the cells rebuilt over the threads ( OpenMP ), after CGAL's parallel triangulation in 3D, its
sequential one in 2D. The binaries are built by `build_baselines.py` ( `cgal/build.sh`, COMPARISONS.md ). `--bin` is the directory that holds them; empty: `cgal/`, `~/.cache/sdot/cgal-bench`, then
`/data/cgal-bench`. A missing binary makes the run fail with the places it looked in.
"""

from errand import Param, bench


CASE_CHOICES = [ "uniform", "lines_voronoi", "lines_equal", "planes_voronoi", "planes_equal" ]

if p := bench( "cgal",
               case   = Param( "uniform", choices = CASE_CHOICES, help = "the case ( see cases.py ); `uniform` uses --dim" ),
               dim    = Param( 2, help = "dimension of the `uniform` case ( 2 or 3 )" ),
               n      = Param( 0, help = "number of seeds ( 0: 1e6 for the uniform ones, the file's size otherwise )" ),
               wscale = Param( 0.0, help = "uniform case: weights in [ -w h^2, w h^2 ] ( 0: Voronoi )" ),
               output = Param( "measures", choices = [ "measures", "adjacency", "triangulation" ], help = "what is computed ( see above )" ),
               threads = Param( 1, help = "TRIANGULATION in 3D: CGAL's threads ( <= 0: the physical cores ); 1 elsewhere" ),
               reps   = Param( 3, help = "timed repetitions, we keep the minimum" ),
               seed   = Param( 0, help = "seed of the uniform draw" ),
               bin    = Param( "", help = "the directory of `power_2d` / `power_3d` ( empty: see above )" ) ):
    import os
    import re
    import subprocess
    import tempfile
    import numpy
    import benchlib
    import cases

    name = f"uniform{ p.dim }d" if p.case == "uniform" else p.case
    pos, w, d = cases.case( name, n = p.n or None, seed = p.seed, wscale = p.wscale )
    n = len( pos )

    here = os.path.dirname( os.path.abspath( __file__ ) )
    prog = f"power_{ d }d"
    dirs = [ p.bin ] if p.bin else [ os.path.join( here, "cgal" ), os.path.join( os.path.expanduser( os.environ.get( "SDOT_CACHE", "~/.cache/sdot" ) ), "cgal-bench" ), "/data/cgal-bench" ]
    exe = next( ( os.path.join( x, prog ) for x in dirs if os.access( os.path.join( x, prog ), os.X_OK ) ), None )
    if exe is None:
        raise FileNotFoundError( f"no `{ prog }` in { ', '.join( dirs ) } ( build it: cgal/build.sh, COMPARISONS.md )" )

    with tempfile.NamedTemporaryFile( "w", suffix = ".txt", delete = False ) as f:
        f.write( f"# { name }, n = { n }\n{ n }\n" )
        numpy.savetxt( f, numpy.column_stack( [ pos, w ] ), fmt = "%.17g" )
        cloud = f.name
    try:
        args = [ exe, "--load", cloud, "--reps", str( p.reps ) ]
        threads = p.threads if p.threads > 0 else benchlib.physical_cores()
        if threads > 1 and not ( p.output == "measures" or ( d == 3 and p.output == "triangulation" ) ):
            raise ValueError( f"no parallel path for --output={ p.output } in { d }D ( CGAL's 2D triangulation is sequential )" )
        if d == 2 and p.output != "measures":
            args.append( f"--{ p.output }" )
        if threads > 1:
            args += [ "--threads", str( threads ) ]
        if d == 3 and p.output == "measures":
            args.append( "--cells" )
        out = subprocess.run( args, capture_output = True, text = True, check = True ).stdout
    finally:
        os.unlink( cloud )
    print( out )

    secs = lambda label: float( re.search( label + r"\s+([0-9.]+) s", out ).group( 1 ) )
    if d == 2:
        t = float( re.search( r"seeds in 2D[^:]*: ([0-9.]+) s", out ).group( 1 ) )
    elif p.output == "triangulation":
        t = secs( "triangulation" )
    elif p.output == "measures":
        t = secs( "WHOLE CHAIN" )
    else:
        t = secs( "triangulation" ) + secs( "adjacency" )

    p.results.update( benchlib.hardware() )
    version = os.path.join( os.path.dirname( exe ), "VERSION" )         # ( build_baselines ); the clone next to them otherwise ( lmo )
    commit = open( version ).read().strip().removeprefix( "cgal " ) if os.path.exists( version ) else subprocess.run(
        [ "git", "-C", os.path.join( os.path.dirname( exe ), "cgal" ), "log", "-1", "--format=%h %ad", "--date=short" ], capture_output = True, text = True ).stdout.strip()
    p.results.update( n = n, dim = d, threads = threads, output = p.output, t_total = t, ns_per_seed = t / n * 1e9, program = prog,
                      cgal_commit = commit )
    print( benchlib.table( [ "case", "n", "output", "CGAL ns/seed" ], [ [ name, n, p.output, f"{ t / n * 1e9 :.0f}" ] ] ) )
