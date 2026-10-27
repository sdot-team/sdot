"""What PRODUCES the catalogue of sdot:  python -m sdot.catalogue

The standard usage of sdot (`provoke`), with the loops to time (`loom.bench_loop`) where the architecture of a kernel is a
choice; the rest -- the loops on the hardware, the collection of the kernels, the timings, the choices, what to leave out of
the wheel -- is `loom.catalogue.build`.

A kernel that is not triggered here is not in the catalogue: it will be compiled on the user's machine (`auto` mode), or
will be missing (`LOOM_KERNELS=catalogue`). The list is therefore a DECISION: what a standard usage does, in 2D and 3D, in
Voronoi and Laguerre, with both kernel floats, the uniform density and an image, and their derivatives.

`level_workloads` is what decides which x86 levels are shipped at all ( see `loom.catalogue.build` ).
"""
from pathlib import Path

import numpy as np


def provoke():
    import loom
    from loom.benchmark import synchronise as wait
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    from sdot.Cell import Cell
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfGaussians import SumOfGaussians

    rng = np.random.default_rng( 0 )
    own = _own_arrays()
    for d in ( 2, 3 ):
        n = 200
        pos = own( rng.uniform( 0.05, 0.95, size = ( n, d ) ) )
        w = own( rng.uniform( -0.3, 0.3, n ) * n ** ( -2.0 / d ) )
        box = box_half_spaces( [ 0 ] * d, [ 1 ] * d )
        for kernel in ( "FP32", "FP64" ):
            for weights in ( None, w ):
                for acc in ( "plain", None ):
                    pd = PowerDiagram( pos, weights = weights, boundaries = box, accelerator = acc, kernel_dtype = kernel )
                    wait( pd.measures.value )
                    pd.moments
                    pd.hessian_rows()
                    pd.cells
                    # derivatives of the measures: with respect to the positions, then the weights
                    _vjp( lambda p: PowerDiagram( p, weights = weights, boundaries = box, accelerator = acc, kernel_dtype = kernel ).measures, pos )
                    if weights is not None:
                        _vjp( lambda q: PowerDiagram( pos, weights = q, boundaries = box, accelerator = acc, kernel_dtype = kernel ).measures, w )
        # an image density, and a sum of Gaussians
        img = Image( values = own( rng.uniform( 0.5, 1.5, size = ( 16, ) * d ) ) )
        pd = PowerDiagram( pos * 16, boundaries = box_half_spaces( [ 0 ] * d, [ 16 ] * d ), distribution = img )
        wait( pd.measures.value )
        pd.moments
        if d == 2:
            sog = SumOfGaussians( positions = pos[ :4 ], sigmas = [ 0.1 ] * 4, weights = [ 1.0 ] * 4 )
            pd = PowerDiagram( pos, boundaries = box, distribution = sog )
            wait( pd.measures.value )
            _vjp( lambda p: PowerDiagram( p, boundaries = box, distribution = sog ).measures, pos )

    # THE TRANSPORT ( `OtProblem.solve` ): one kernel per ( dimension, storage, density ) -- on CPU only,
    # where the solver lives ( `sdot/sdotplan/` ); an image and Gaussians in 2D, an image in 3D
    from sdot.OtProblem import Iterative, OtProblem, Tuning
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    if loom.resolved_device().is_cpu:
        for d in ( 2, 3 ):
            pos = own( rng.uniform( 0.05, 0.95, size = ( 100, d ) ) )
            img = Image( values = own( rng.uniform( 0.5, 1.5, size = ( 8, ) * d ) ), origin = [ 0.0 ] * d, pixel_frame = ( np.eye( d ) / 8 ).tolist() )
            for acc in ( None, "plain" ):
                plan = OtProblem( SumOfDiracs( pos ), img ).solve( Iterative( max_iter = 20, tuning = Tuning( accelerator = acc ) ) )
                plan.cost_and_position_grad()
            if d == 2:
                sog = SumOfGaussians( positions = pos[ :4 ], sigmas = [ 0.1 ] * 4, weights = [ 1.0 ] * 4 )
                OtProblem( SumOfDiracs( pos ), sog ).solve( Iterative( max_iter = 20 ) )

    # the cell alone, what `Cell` exposes
    for d in ( 2, 3 ):
        c = Cell.make_hypercube( d, [ 0 ] * d, np.eye( d ).tolist() )
        c.cut( [ 1.0 ] * d, 1.0 )
        wait( c.measure.value )

    documented()
    timed()


# What the documentation shows ( docs/snippets ): a user starts with these, so their kernels are in the catalogue -- whatever
# settings they use by default. `scripts/validate_wheel.py` runs the same files on the installed wheel with
# `LOOM_KERNELS=catalogue`, so a kernel missing here fails the release instead of compiling on a user's machine.
DOCUMENTED = [ "hello_numpy.py", "hello_jax.py", "hello_torch.py", "hello_cupy.py", "plan.py", "ot_solve.py", "ot_problem.py",
               "distributions.py", "distributions_1d.py", "duplicates_plan.py" ]
SNIPPETS = Path( __file__ ).resolve().parents[ 2 ] / "docs" / "snippets"


def documented():
    """Run the snippets of `DOCUMENTED`, as a user would ( their numpy inputs go to the framework in force ), in a scratch
    directory ( one writes a page ). A snippet of a framework that is not installed, or that does not run on this device ( the
    numpy driver on a card ), is skipped; the `hello_<framework>` of each framework runs only under that framework."""
    import contextlib, io, runpy, tempfile, os
    from loom.drivers.Framework import UnsupportedDevice
    from loom.drivers import framework_defaults
    mine = str( framework_defaults.framework() )
    if not SNIPPETS.is_dir():
        raise SystemExit( f"sdot.catalogue: { SNIPPETS } is missing -- the catalogue is built from a checkout, with its docs" )
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir( scratch )
        try:
            for name in DOCUMENTED:
                if name.startswith( "hello_" ) and name != f"hello_{ mine }.py":
                    continue                            # a framework's own hello: a user of that framework runs it
                try:
                    with contextlib.redirect_stdout( io.StringIO() ):
                        runpy.run_path( str( SNIPPETS / name ), run_name = "__main__" )
                except ( ModuleNotFoundError, UnsupportedDevice ) as e:
                    print( f"   { name }: skipped ( { e } )", flush = True )
        finally:
            os.chdir( here )


def timed():
    """The kernels whose architecture is a CHOICE: run at a size where they dominate, timed on every architecture the
    machine can behave as (`loom.bench_loop`; a plain loop that runs once when nothing is being built)."""
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    from sdot.distributions.Image import Image
    from sdot.SdotPlan1d import SdotPlan1d
    from sdot.distributions.SumOfDiracs1d import SumOfDiracs1d
    import loom
    from loom.benchmark import synchronise as wait
    rng = np.random.default_rng( 0 )
    n = 200000
    own = _own_arrays()
    pos = own( rng.uniform( 0.05, 0.95, size = ( n, 3 ) ) )
    box = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 )
    for _ in loom.bench_loop( "power_diagram_measures" ):
        wait( PowerDiagram( pos, boundaries = box, accelerator = None ).measures.value )
    for _ in loom.bench_loop( "power_diagram_moments" ):
        PowerDiagram( pos, boundaries = box, accelerator = None ).moments

    m = 1000000
    src = SumOfDiracs1d( positions = own( np.sort( rng.uniform( 0, 1, m ) ) ) )
    img = Image( values = own( rng.uniform( 0.5, 1.5, 20000 ) ) )
    for _ in loom.bench_loop( "update_outputs_SdotPlan1d" ):
        float( SdotPlan1d( src, img ).cost )


def _own_arrays():
    """The arrays of the framework in force (Torch does not mix its tensors with numpy's; Jax does)."""
    from loom.drivers import framework_defaults
    import loom
    return ( lambda a: a ) if framework_defaults.framework_name() == "jax" else loom.array


def _vjp( f, x ):
    """A backward pass, to trigger the `bwd` kernel of `f` (same path as `check_grad`). A framework without autodiff (numpy, cupy)
    has none, hence no such kernel."""
    import loom
    probe = f( x )
    dense_shape = tuple( probe.shape )
    crop = lambda t: t.raw[ tuple( slice( 0, s ) for s in dense_shape ) ]
    try:
        out, pullback = loom.vjp( lambda v: crop( f( v ) ), x )
    except NotImplementedError:
        return
    pullback( loom.random( out.shape, seed = 0 ) )


def level_workloads():
    """What decides which x86 levels the wheels carry ( `loom.catalogue.build( levels = ... )` ): the operations of a standard
    use, at sizes where the kernels dominate, equal weights. Each is timed at every level; a level that gains too little on
    them is not shipped, and its machines run the poorer one kept ( on lmo, 2026-10-20, before the `cells` workloads: v3 0.94 of the
    baseline, v4 0.98, v2 1.00 ). Built when called: the arrays are those of the framework in force."""
    from loom.benchmark import synchronise as wait
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.distributions.SumOfDiracs1d import SumOfDiracs1d
    from sdot.SdotPlan1d import SdotPlan1d
    from sdot.OtProblem import Iterative, OtProblem
    rng = np.random.default_rng( 0 )
    own = _own_arrays()
    p2, p3 = own( rng.uniform( 0.01, 0.99, ( 1000000, 2 ) ) ), own( rng.uniform( 0.01, 0.99, ( 200000, 3 ) ) )
    w2 = own( rng.uniform( -0.3, 0.3, 1000000 ) / 1000000 )
    b2, b3, b256 = box_half_spaces( [ 0 ] * 2, [ 1 ] * 2 ), box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 ), box_half_spaces( [ 0 ] * 2, [ 256 ] * 2 )
    img = own( rng.uniform( 0.5, 1.5, ( 256, 256 ) ) )
    s1, i1 = own( np.sort( rng.uniform( 0, 1, 1000000 ) ) ), own( rng.uniform( 0.5, 1.5, 20000 ) )
    ot = own( rng.uniform( 0.05, 0.95, ( 20000, 2 ) ) )
    return {
        # THE CELLS, the diagram prepared once per level: what a solve repeats, and what the level changes ( the 2D register
        # kernel is `SimdVec<double,8>`: x86-64 has no register permute for it, AVX2 two ymm, AVX-512 one zmm ). A new
        # diagram per call would time the tree as well, which the level barely moves and which then hides the cells.
        "cells 2D 1e6 Voronoi":     _cells( p2, None, b2 ),
        "cells 2D 1e6 Laguerre":    _cells( p2, w2, b2 ),
        "cells 3D 2e5":             _cells( p3, None, b3 ),
        "measures 2D 1e6 Laguerre": lambda: wait( PowerDiagram( p2, weights = w2, boundaries = b2 ).measures.value ),
        "measures 3D 2e5":          lambda: wait( PowerDiagram( p3, boundaries = b3 ).measures.value ),
        "moments 3D 2e5":           lambda: PowerDiagram( p3, boundaries = b3 ).moments,
        "hessian 2D 1e6":           lambda: PowerDiagram( p2, weights = w2, boundaries = b2 ).hessian_rows(),
        "image measures 2D 1e6":    lambda: wait( PowerDiagram( p2 * 256, boundaries = b256, distribution = Image( values = img ) ).measures.value ),
        "SdotPlan1d 1e6":           lambda: float( SdotPlan1d( SumOfDiracs1d( positions = s1 ), Image( values = i1 ) ).cost ),
        "OT solve 2D 2e4 image":    lambda: OtProblem( SumOfDiracs( ot ), Image( values = img, origin = [ 0.0, 0.0 ],
                                                       pixel_frame = ( np.eye( 2 ) / 256 ).tolist() ) ).solve( Iterative( max_iter = 30 ) ),
    }


def _cells( positions, weights, boundaries ):
    """a workload that times the measures of ONE diagram per level: built on the first call at a level ( the warm-up of
    `_time_levels`, which also compiles ), then measured again with the weights set anew ( what a Newton step does ).
    Keyed by the level the device behaves as: a diagram lays out its buffers for the level it was built at."""
    import loom
    from loom.benchmark import synchronise as wait
    from sdot.PowerDiagram import PowerDiagram
    built = {}
    def run():
        level = loom.resolved_device().catalogue_target()
        if level not in built:
            built[ level ] = PowerDiagram( positions, weights = weights, boundaries = boundaries )
        pd = built[ level ]
        if weights is not None:
            pd.weights = weights
        return wait( pd.measures.value )
    return run


if __name__ == "__main__":
    import loom
    loom.catalogue.build( "sdot", provoke, levels = level_workloads )
