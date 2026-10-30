"""GEOGRAM ON THE SAME CLOUDS: Bruno Lévy's restricted power diagrams and his KMT Newton ( exploragram's
`OptimalTransportMap2d / 3d` ), through the driver `geogram/geogram_ot.cpp`, on the clouds `bench_diagram` / `bench_newton` use,
a constant density on the unit square / cube.

    errand run --env lmo-numpy -x 'sdot-ffi/bench/bench_geogram.py::*' --case=uniform --dim=3 --level=diagram --threads=1,8
    errand run --env lmo-numpy -x 'sdot-ffi/bench/bench_geogram.py::*' --case=planes_voronoi --level=newton --threads=8

`--level`:
  diagram   the measure of every cell at the given weights ( the file's for the `equal` cases, `--wscale` for the uniform
            ones ): the regular triangulation of the lifted seeds ( 2D `BPOW2d`, sequential; 3D `PDEL`, parallel ) then the
            cells restricted to a simplicial mesh of the box ( parallel over the parts of the mesh ). Min of `--reps`, after a
            warm-up. `ns_per_seed` is the whole thing, the two halves are recorded apart.
  adjacency the neighbours of every cell with the facet terms, what sdot's `hessian_rows` gives ( the `adjacency` job ):
            exploragram's `compute_P1_Laplacian`, the triangulation then the restricted cells assembled in a sparse matrix,
            at the same weights as `diagram`. Min of `--reps`, after a warm-up.
  newton    exploragram's `optimize_full_Newton` from zero weights: Jacobi CG to 1e-3 ( its default ), the KMT step
            ( smallest cell >= half the smallest at the start, |g| decreasing ). The driver copies the loop with ONE change, the
            stop: sdot's `max |m - nu| / nu <= --rtol` instead of geogram's `|m - nu|_2 <= sqrt( n ) eps nu`. `--multilevel=yes`:
            Lévy's multilevel scheme ( BRIO levels of ratio 1/8, the coarse levels to geogram's own 1 %, the last one to rtol ).

`--check=yes` ( the default ) recomputes with sdot, on the same points: the measures ( diagram: max relative difference ), or
the neighbours ( adjacency: the seeds whose number of neighbours differs ), or the residual at the weights geogram returns ( newton: COMPARISONS.md says the stop is checked by sdot, whatever the solver
says ). The check is not timed.

The domain is the unit box cut in `--grid` sub-boxes per side, 2 triangles / 6 tetrahedra each ( 0: the smallest grid giving
4 simplices per thread -- geogram parallelises the cells over a partition of the mesh, and each simplex boundary is one
more cut ). The driver and geogram are not built here: `geogram/build.sh` ( COMPARISONS.md, "geogram" ); `--bin` is the
directory of `geogram_ot` ( empty: `geogram/`, `~/.cache/sdot/geogram-bench`, then `/data/geogram-bench` ).
"""

from errand import Param, bench


CASE_CHOICES = [ "uniform", "lines_voronoi", "lines_equal", "planes_voronoi", "planes_equal" ]

if p := bench( "geogram",
               case       = Param( "uniform", choices = CASE_CHOICES, help = "the case ( see cases.py ); `uniform` uses --dim" ),
               dim        = Param( 2, help = "dimension of the `uniform` case ( 2 or 3 )" ),
               n          = Param( 0, help = "number of seeds ( 0: the campaign's: uniform 1e6 for a diagram, 1e5 for Newton; the file's size otherwise )" ),
               wscale     = Param( 0.0, help = "uniform case, diagram: weights in [ -w h^2, w h^2 ] ( 0: Voronoi )" ),
               level      = Param( "diagram", choices = [ "diagram", "adjacency", "newton" ], help = "one diagram with its measures, its neighbours, or a Newton solve from zero weights" ),
               threads    = Param( -1, help = "geogram's threads ( and OpenMP's, for its CG ); <= 0: the physical cores" ),
               grid       = Param( 0, help = "sub-boxes per side of the domain mesh ( 0: see above )" ),
               multilevel = Param( "no", choices = [ "no", "yes" ], help = "NEWTON: Lévy's multilevel scheme" ),
               linsolve_eps = Param( 0.0, help = "NEWTON: relative tolerance of the Jacobi CG ( 0: exploragram's default, 1e-3 )" ),
               rtol       = Param( 1e-6, help = "NEWTON: stop on max|m - nu| / nu <= rtol" ),
               max_iter   = Param( 100, help = "NEWTON: steps, at most" ),
               reps       = Param( 3, help = "DIAGRAM, ADJACENCY: timed repetitions, we keep the minimum; NEWTON: runs, we keep the fastest" ),
               seed       = Param( 0, help = "seed of the uniform draw" ),
               check      = Param( "yes", choices = [ "yes", "no" ], help = "compare with sdot ( see above )" ),
               bin        = Param( "", help = "the directory of `geogram_ot` ( empty: see above )" ) ):
    import os
    import re
    import subprocess
    import tempfile
    import numpy
    import benchlib
    import cases

    threads = p.threads if p.threads > 0 else benchlib.physical_cores()
    name = f"uniform{ p.dim }d" if p.case == "uniform" else p.case
    n_arg = p.n or ( 100_000 if p.level == "newton" and p.case == "uniform" else None )
    pos, w, d = cases.case( name, n = n_arg, seed = p.seed, wscale = p.wscale )
    if p.level == "newton":
        w = numpy.zeros( len( pos ) )               # from Voronoi ( the `equal` weights are the solution )
    n = len( pos )

    here = os.path.dirname( os.path.abspath( __file__ ) )
    dirs = [ p.bin ] if p.bin else [ os.path.join( here, "geogram" ), os.path.join( os.path.expanduser( os.environ.get( "SDOT_CACHE", "~/.cache/sdot" ) ), "geogram-bench" ), "/data/geogram-bench" ]
    exe = next( ( os.path.join( x, "geogram_ot" ) for x in dirs if os.access( os.path.join( x, "geogram_ot" ), os.X_OK ) ), None )
    if exe is None:
        raise FileNotFoundError( f"no `geogram_ot` in { ', '.join( dirs ) } ( build it: geogram/build.sh, COMPARISONS.md )" )
    version_file = os.path.join( os.path.dirname( exe ), "VERSION" )
    version = open( version_file ).read().strip().replace( "\n", ", " ) if os.path.exists( version_file ) else "?"

    tmp = tempfile.mkdtemp()
    cloud, out_file = os.path.join( tmp, "cloud.bin" ), os.path.join( tmp, "out.bin" )
    with open( cloud, "wb" ) as f:
        numpy.array( [ n, d ], numpy.int64 ).tofile( f )
        numpy.ascontiguousarray( pos, numpy.float64 ).tofile( f )
        numpy.ascontiguousarray( w, numpy.float64 ).tofile( f )
    args = [ exe, "--load", cloud, "--mode", p.level, "--threads", str( threads ), "--grid", str( p.grid ), "--out", out_file ]
    if p.level != "newton":
        args += [ "--reps", str( p.reps ) ]
    else:
        args += [ "--rtol", str( p.rtol ), "--max-iter", str( p.max_iter ), "--multilevel", "1" if p.multilevel == "yes" else "0" ]
        if p.linsolve_eps > 0:
            args += [ "--linsolve-eps", str( p.linsolve_eps ) ]

    env = dict( os.environ, OMP_NUM_THREADS = str( threads ) )
    runs = []
    try:
        for r in range( 1 if p.level != "newton" else p.reps ):     # the diagram repeats inside the driver
            res = subprocess.run( args, capture_output = True, text = True, env = env )
            if res.returncode != 0:
                raise RuntimeError( f"geogram_ot failed ( { res.returncode } ):\n{ res.stdout }\n{ res.stderr }" )
            runs.append( ( res.stdout, res.stderr, numpy.fromfile( out_file, numpy.float64 ) ) )
    finally:
        for f in ( cloud, out_file ):
            if os.path.exists( f ):
                os.unlink( f )
        os.rmdir( tmp )

    num = lambda out, key: float( re.search( rf"^{ key }: ([-+0-9.eE]+)", out, re.M ).group( 1 ) )
    if p.level == "newton":
        runs.sort( key = lambda r: num( r[ 0 ], "t_newton" ) )
    out, err, values = runs[ 0 ]
    print( out )
    if err.strip():
        print( err )
    head = re.search( r"threads (\d+) grid (\d+) simplices (\d+)", out )

    p.results.update( benchlib.hardware() )
    p.results.update( n = n, dim = d, case = name, level = p.level, threads = int( head.group( 1 ) ), grid = int( head.group( 2 ) ),
                      simplices = int( head.group( 3 ) ), geogram = version )

    sdot_check = None
    if p.check == "yes":
        from sdot import PowerDiagram, box_half_spaces
        weights = w if p.level != "newton" else values
        pd = PowerDiagram( pos, weights = weights, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ), kernel_dtype = "FP64" )
        if p.level == "adjacency":
            nb, ids, _ = pd.hessian_rows()
            nb_sdot = numpy.sum( ( ids >= 0 ) & ( numpy.arange( ids.shape[ 1 ] ) < nb[ :, None ] ), axis = 1 )   # the domain's are negative
            sdot_check = int( numpy.sum( nb_sdot != numpy.rint( values ).astype( int ) ) )
            p.results.update( max_diff_sdot = sdot_check, mean_neighbours_sdot = float( nb_sdot.mean() ) )
            print( f"  neighbours against sdot: { sdot_check } seeds of { n } with another count ( mean { values.mean() :.3f} in geogram, "
                   f"{ nb_sdot.mean() :.3f} in sdot )" )
        elif p.level == "diagram":
            m_sdot = numpy.asarray( pd.measures, numpy.float64 )
            ok = m_sdot > 0
            rel = numpy.abs( values[ ok ] - m_sdot[ ok ] ) / m_sdot[ ok ]
            sdot_check = float( numpy.max( numpy.abs( values - m_sdot ) ) * n )     # in units of the mean cell, 1 / n
            p.results.update( max_diff_sdot = sdot_check, max_rel_diff_sdot = float( rel.max() ), median_rel_diff_sdot = float( numpy.median( rel ) ),
                              empty_sdot = int( numpy.sum( ~ok ) ), empty_geogram = int( numpy.sum( values <= 0 ) ) )
            worst = numpy.argmax( numpy.abs( values - m_sdot ) )
            print( f"  measures against sdot: max |m_geogram - m_sdot| * n = { sdot_check :.2e} ( seed { worst }: { values[ worst ] :.6e} against { m_sdot[ worst ] :.6e} ); "
                   f"relative to the cell: max { rel.max() :.2e}, median { numpy.median( rel ) :.2e}; "
                   f"{ int( numpy.sum( ~ok ) ) } empty cells in sdot, { int( numpy.sum( values <= 0 ) ) } in geogram" )
        else:
            sdot_check = float( numpy.max( numpy.abs( numpy.asarray( pd.measures, numpy.float64 ) * n - 1 ) ) )
            p.results.update( residual_sdot = sdot_check )
            print( f"  residual measured by sdot at geogram's weights: max |m - nu| / nu = { sdot_check :.2e}" )

    if p.level == "diagram":
        t = num( out, "diagram" )
        p.results.update( t_total = t, ns_per_seed = t / n * 1e9, t_triangulation = num( out, "triangulation" ), t_rvd = num( out, "rvd" ),
                          sum_measures = num( out, "sum_measures" ) )
        print( benchlib.table( [ "case", "n", "threads", "geogram ns/seed", "triangulation", "cells" ],
                               [ [ name, n, threads, f"{ t / n * 1e9 :.0f}", f"{ num( out, 'triangulation' ) / n * 1e9 :.0f}",
                                   f"{ num( out, 'rvd' ) / n * 1e9 :.0f}" ] ] ) )
    elif p.level == "adjacency":
        t = num( out, "adjacency" )
        p.results.update( t_total = t, ns_per_seed = t / n * 1e9, mean_neighbours = num( out, "mean_neighbours" ) )
        print( benchlib.table( [ "case", "n", "threads", "geogram ns/seed", "mean neighbours" ],
                               [ [ name, n, threads, f"{ t / n * 1e9 :.0f}", f"{ num( out, 'mean_neighbours' ) :.2f}" ] ] ) )
    else:
        keys = [ "iterations", "triangulations", "rvd_passes", "trials", "cg_iterations", "t_triangulation", "t_rvd", "t_rvd_hessian", "t_linear" ]
        p.results.update( { k: num( out, k ) for k in keys } )
        p.results.update( t_total_wall = num( out, "t_newton" ), converged = bool( num( out, "converged" ) ), residual = num( out, "residual" ),
                          diagrams = int( num( out, "triangulations" ) ), multilevel = p.multilevel, linsolve_eps = p.linsolve_eps or 1e-3, rtol = p.rtol,
                          t_all_runs = [ num( r[ 0 ], "t_newton" ) for r in runs ] )
        if p.multilevel == "yes":
            p.results.update( t_coarse_levels = num( out, "coarse_levels" ) )
        print( benchlib.table( [ "case", "n", "threads", "converged", "seconds", "iterations", "diagrams", "residual ( sdot )" ],
                               [ [ name, n, threads, p.results[ "converged" ], f"{ p.results[ 't_total_wall' ] :.2f}", int( num( out, "iterations" ) ),
                                   int( num( out, "triangulations" ) ), f"{ sdot_check :.1e}" if sdot_check is not None else "-" ] ] ) )
