"""ONE DIAGRAM OF DIRACS ON A GRID, by the grid's dedicated geometry ( `grid/GridNewton2D.h`, `Evaluator` ): the cells of the
N x N centres at fixed weights, their areas and the weights of the hessian ( Lebesgue on the unit square ), what one Newton
step of the dedicated solver computes. On the grid cases of `cases.py`, those `bench_diagram` / `bench_cgal` / `bench_geogram`
take as clouds.

    errand run --env lmo-numpy 'sdot-ffi/bench/bench_grid_diagram.py::*' --case=grid_sines,grid_sharp --n=1048576 --threads=1 --method=zones

`--method`:
  zones   the zone certificate ( `use_zones`, the default ): ring 1 in SIMD, a table of the excess of the cell areas per zone,
          the cells of the bad zones rebuilt with more seeds until no zone is bad -- exact, whatever the weights
  cert    the former local certificates ( ring 1, ring 2 on the cells it flags, a cross check, the exact clipping with the
          radius of security of max psi )

The time is that of `Evaluator::run` ( geometry, areas, faces ), the smallest of `--reps` after a warm-up call; `t_geometry`
that of the geometry alone. The sum of the areas is checked against the domain.
"""

from errand import Param, bench

import benchlib

CASES = [ "grid_voronoi", "grid_sines", "grid_sharp" ]

if p := bench( "grid_diagram",
               case    = Param( "grid_sines", choices = CASES, help = "the case ( see cases.py )" ),
               n       = Param( 1024 ** 2, help = "number of seeds, a square" ),
               method  = Param( "zones", choices = [ "zones", "cert" ], help = "the certificate ( see above; not `path`: errand leaves it out of the run names )" ),
               threads = Param( 1, help = "workers ( 0: all the cores, -1: the physical ones )" ),
               pin     = Param( "yes", choices = [ "yes", "no", "env" ], help = "pin worker w to core w" ),
               reps    = Param( 3, help = "timed repetitions, the minimum is kept" ),
               kernels = Param( "catalogue", choices = [ "catalogue", "compile" ], help = "`compile`: compiled for this machine" ),
               trace   = Param( 0, help = "1: the phases of one diagram on stdout ( the card )" ),
               device  = Param( "cpu", choices = [ "cpu", "default" ], help = "`default`: the card's port ( `gpu/GridNewton2D.cuh::diagram_call` )" ) ):
    benchlib.set_device( p.device )                # BEFORE importing loom
    benchlib.set_kernels( p.kernels )
    benchlib.set_threads( p.threads, p.pin )
    import numpy
    import loom
    import sdot                                    # noqa: F401 ( it declares the externals of its kernels: eigen )
    from loom.compilation.FfiCode import FfiCode
    from loom.tensor import Axis, RealTensor, ShapeVar
    import cases

    pos, w, d = cases.case( p.case, n = p.n )
    n = len( pos )
    N = int( round( n ** 0.5 ) )
    psi = w * N * N                                # ( index units: psi = w / h^2 )

    def axis( m, name ):
        return Axis( ShapeVar( int( m ) ), name = name )
    nb_times = 6
    masses = RealTensor[ axis( n, "num_seed" ) ]()
    times = RealTensor[ axis( nb_times, "num_time" ) ]()
    opts = numpy.asarray( [ N, 1.0 if p.method == "zones" else 0.0, max( p.reps, 1 ) ], dtype = numpy.float64 )
    print( "  " + benchlib.env_report() )
    benchlib.start_clock()
    card = p.device != "cpu"
    if card:
        opts = numpy.asarray( [ N, 1.0 if p.method == "zones" else 0.0, max( p.reps, 1 ), float( p.trace ) ], dtype = numpy.float64 )
        loom.ffi_call(
            "sdot_bench_grid_diagram_card",
            FfiCode.inline( "sdot::gpu2d::grid::diagram_call( queue, args.allocator, args.inputs.psi, args.inputs.options, args.outputs.masses, args.outputs.times );",
                            includes = [ "sdot/gpu/GridNewton2D.cuh" ], allocator = True ),
            psi = RealTensor[ axis( n, "num_seed" ) ]( numpy.ascontiguousarray( psi, dtype = numpy.float64 ) ),
            options = RealTensor[ axis( len( opts ), "num_opt" ) ]( opts ),
            masses = loom.out( masses ), times = loom.out( times ),
            has_dynamic_capacity = False,
        )
    else: loom.ffi_call(
        "sdot_bench_grid_diagram",
        FfiCode.inline( includes = [ "sdot/grid/GridNewton2D.h", "chrono" ], sources = [ "sdot/sdotplan/Linear.cpp" ], externals = [ "eigen" ], code = "\n".join( [
            "auto &inputs = args.inputs; auto &outputs = args.outputs;",
            "const int N = int( inputs.options( 0 ) ), reps = int( inputs.options( 2 ) ); const bool zones = inputs.options( 1 ) != 0;",
            "const SI n = SI( N ) * N;",
            "std::vector<double> psi( n ); for ( SI i = 0; i < n; ++i ) psi[ i ] = double( inputs.psi( i ) );",
            "sdot::grid::Par par( queue );",
            "sdot::grid::Evaluator<sdot::grid::UnitCells> ev( N, sdot::grid::UnitCells{}, par );",
            "ev.use_zones = zones;",
            "ev.run( psi );                                   // ( warm-up: the list of the exact path grows there if needed )",
            "auto now = [] { return std::chrono::duration<double>( std::chrono::steady_clock::now().time_since_epoch() ).count(); };",
            "double t_run = 1e300, t_geo = 1e300;",
            "for ( int r = 0; r < reps; ++r ) { const double t0 = now(); ev.run( psi ); t_run = std::min( t_run, now() - t0 ); }",
            "for ( int r = 0; r < reps; ++r ) { const double t0 = now(); ev.geometry( psi ); t_geo = std::min( t_geo, now() - t0 ); }",
            "ev.run( psi );",
            "for ( SI i = 0; i < n; ++i ) outputs.masses( i ) = ev.mass[ i ];",
            "outputs.times( 0 ) = t_run; outputs.times( 1 ) = t_geo; outputs.times( 2 ) = ev.n_flag1; outputs.times( 3 ) = ev.n_exact;",
            "outputs.times( 4 ) = ev.exhausted; outputs.times( 5 ) = ev.n_rounds;",
        ] ) ),
        psi = RealTensor[ axis( n, "num_seed" ) ]( numpy.ascontiguousarray( psi, dtype = numpy.float64 ) ),
        options = RealTensor[ axis( len( opts ), "num_opt" ) ]( opts ),
        masses = loom.out( masses ), times = loom.out( times ),
        has_dynamic_capacity = False,
    )
    from loom.tensor.host import to_host
    t = numpy.asarray( to_host( getattr( times, "raw", times ) ), dtype = float )
    m = numpy.asarray( to_host( getattr( masses, "raw", masses ) ), dtype = float )
    if card:                                       # ( the card: geometry, geometry + areas, flagged, rebuilt, rounds, exhausted )
        t = numpy.asarray( [ t[ 1 ], t[ 0 ], t[ 2 ], t[ 3 ], t[ 5 ], t[ 4 ] ] )
    err = abs( m.sum() - n ) / n
    p.results.update( benchlib.hardware() )
    threads = p.threads if p.threads > 0 else benchlib.physical_cores()
    p.results.update( n = n, dim = 2, threads = threads, device = p.device, output = "measures", path = p.method, t_total = float( t[ 0 ] ), t_geometry = float( t[ 1 ] ),
                      ns_per_seed = float( t[ 0 ] / n * 1e9 ), ns_per_seed_geometry = float( t[ 1 ] / n * 1e9 ), flagged = int( t[ 2 ] ),
                      exact_path = int( t[ 3 ] ), exhausted = bool( t[ 4 ] ), rounds = int( t[ 5 ] ), mass_error = float( err ) )
    if err > 1e-9 or t[ 4 ]:
        raise RuntimeError( f"the areas sum to { m.sum() } for a domain of { n } ( exhausted: { bool( t[ 4 ] ) } )" )
    print( benchlib.table( [ "case", "n", "path", "ns/seed", "geometry ns/seed", "rebuilt", "exact path" ],
                           [ [ p.case, n, p.method, f"{ t[ 0 ] / n * 1e9 :.0f}", f"{ t[ 1 ] / n * 1e9 :.0f}", int( t[ 2 ] ), int( t[ 3 ] ) ] ] ) )
