"""THE DIRACS OF A GRID ON THE CARD ( `gpu/GridNewton2D.cuh`, `SdotPlanNd._build_grid_card` ): the dedicated solver's port, taken by
default on a CUDA device. The references: the CPU's dedicated solver on the same problem ( a subprocess on `LOOM_DEVICE=cpu`: one
process has one device ), and the card's generic Newton on the positions of the same diracs ( `grid_solver = "cells"`: on the
card, the grid goes to its positions ) for the densities it integrates.

The images: constant pixels, and their bilinear interpolations ( `SplineGrid( degree = 1, interpolate = True )`, an `Image( degree =
1 )` ) -- the stripes of contrast 10 are a hard case for constant pixels ( coarse to fine fails, on the CPU as here ), not for their
interpolation.

    errand run --no-queue --env lmo-jax 'sdot-ffi/tests/test_SumOfDiracsOnAGrid_card.py::*'
"""

from errand import test


def _host( x ):
    import numpy
    from loom.tensor.host import to_host
    return numpy.asarray( to_host( getattr( x, "value", x ) ), dtype = float )


def _on_card():
    import loom
    from errand import skip
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the card's grid solver only exists on a CUDA device" )


def _grid_masses( N ):
    import numpy
    u = ( numpy.arange( N ) + 0.5 ) / N
    X, Y = numpy.meshgrid( u, u, indexing = "ij" )
    m = numpy.exp( -( ( X - 0.35 ) ** 2 + ( Y - 0.6 ) ** 2 ) / 0.08 ) + 0.3
    return m.ravel() / m.sum()


def _targets():
    import numpy
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.distributions.SplineGrid import SplineGrid
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    x, y = ( numpy.arange( 40 ) + 0.5 ) / 40, ( numpy.arange( 24 ) + 0.5 ) / 24
    u = ( numpy.arange( 24 ) + 0.5 ) / 24
    smooth = 1 + 0.5 * numpy.sin( 5 * u[ :, None ] ) * numpy.cos( 3 * u[ None, : ] )
    stripes = 1 + 9.0 * ( numpy.add.outer( numpy.arange( 64 ), numpy.arange( 64 ) ) % 7 < 3 )
    return ( ( "uniform density", lambda: Box( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ),
             ( "image", lambda: Image( values = smooth ) ),
             ( "image", lambda: Image( values = 1 + 0.6 * numpy.cos( 4 * x[ :, None ] + 2 * y[ None, : ] ), frame = [ 1.0, 1.0 ] ) ),
             # ( pixels on the edges of the cells: 64 x 64 on a 64 x 64 grid, a contrast of 10 )
             ( "image stripes", lambda: Image( values = stripes ) ),
             ( "image of degree 1", lambda: SplineGrid( smooth, degree = 1, interpolate = True ) ),
             ( "image of degree 1 stripes", lambda: SplineGrid( stripes, degree = 1, interpolate = True ) ),
             ( "gaussians", lambda: SumOfGaussians( numpy.array( [ [ 0.3, 0.4 ], [ 0.7, 0.6 ] ] ), numpy.array( [ 0.2, 0.25 ] ),
                                                    weights = numpy.array( [ 1.0, 0.7 ] ), support_box = ( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ) ),
             ( "gaussians", lambda: SumOfGaussians( numpy.array( [ [ 0.5, 0.5 ] ] ), numpy.array( [ 0.08 ] ), support_box = ( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ) ) )


if test( "the_card_solves_the_grid_as_the_generic_newton" ):
    import numpy
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem, Tuning
    _on_card()
    m = _grid_masses( 64 )
    for kind, target in _targets():
        if "degree 1" in kind:
            continue                                     # ( the card's generic Newton does not integrate a polynomial image )
        ref = OtProblem( SumOfDiracsOnAGrid( ( 64, 64 ), weights = m ), target() ).solve( Iterative( tol = 1e-12, tuning = Tuning( grid_solver = "cells" ) ) )
        assert ref.converged and "grid_solver" not in ref.stats, ( kind, ref.stats )
        # ( the stripes of contrast 10: coarse to fine from 32 fails on the CPU's dedicated solver as here, the same iterations )
        for n0 in ( ( 64, ) if "stripes" in kind else ( 64, 32 ) ):
            p = OtProblem( SumOfDiracsOnAGrid( ( 64, 64 ), weights = m ), target() ).solve( Iterative( tol = 1e-12, tuning = Tuning( grid_n0 = n0 ) ) )
            assert p.converged and p.stats[ "grid_solver" ] == f"dedicated card ( { kind.replace( " stripes", "" ) } )", ( kind, p.stats )
            w, wr = _host( p.weights ), _host( ref.weights )
            assert numpy.allclose( w - w[ 0 ], wr - wr[ 0 ], atol = 1e-9 ), ( kind, n0, numpy.abs( ( w - w[ 0 ] ) - ( wr - wr[ 0 ] ) ).max() )
            assert numpy.allclose( _host( p.cell_masses ), _host( ref.cell_masses ), atol = 1e-11 ), ( kind, n0 )
            assert numpy.allclose( _host( p.barycenters ), _host( ref.barycenters ), atol = 1e-8 ), ( kind, n0, numpy.abs( _host( p.barycenters ) - _host( ref.barycenters ) ).max() )
            assert numpy.isclose( float( _host( p.cost ) ), float( _host( ref.cost ) ), rtol = 1e-8 ), ( kind, n0 )

_CPU_SOLVE = """
import pickle, sys, numpy
sys.path.insert( 0, sys.argv[ 3 ] )
from test_SumOfDiracsOnAGrid_card import _targets, _grid_masses
from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
from sdot.OtProblem import Iterative, OtProblem, Tuning
res = []
for q, n0 in pickle.load( open( sys.argv[ 1 ], "rb" ) ):
    p = OtProblem( SumOfDiracsOnAGrid( ( 64, 64 ), weights = _grid_masses( 64 ) ), _targets()[ q ][ 1 ]() ).solve(
        Iterative( tol = 1e-12, on_failure = "ignore", tuning = Tuning( grid_solver = "dedicated", grid_n0 = n0 ) ) )
    res.append( dict( weights = numpy.asarray( p.weights ).reshape( -1 ), masses = numpy.asarray( p.cell_masses ).reshape( -1 ),
                      bary = numpy.asarray( p.barycenters ), cost = float( p.cost ), stats = p.stats ) )
pickle.dump( res, open( sys.argv[ 2 ], "wb" ) )
"""


def _cpu_solves( cases ):
    """the CPU's dedicated solves of `cases` ( `( target index, n0 )` ), in a subprocess on the CPU device"""
    import os, pickle, subprocess, sys, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fi, fo = os.path.join( tmp, "in.pkl" ), os.path.join( tmp, "out.pkl" )
        pickle.dump( cases, open( fi, "wb" ) )
        env = dict( os.environ, LOOM_DEVICE = "cpu", JAX_PLATFORMS = "cpu" )
        r = subprocess.run( [ sys.executable, "-c", _CPU_SOLVE, fi, fo, os.path.dirname( os.path.abspath( __file__ ) ) ], env = env,
                            capture_output = True, text = True )
        assert r.returncode == 0, r.stdout[ -3000: ] + r.stderr[ -3000: ]
        return pickle.load( open( fo, "rb" ) )


if test( "the_card_solves_the_grid_as_the_cpu" ):
    import numpy
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem, Tuning
    _on_card()
    m = _grid_masses( 64 )
    targets = _targets()
    # ( the stripes, constant or interpolated: their contrast is at the scale of the cells, coarse to fine from 32 fails on both sides )
    cases = [ ( q, n0 ) for q, ( kind, _ ) in enumerate( targets ) for n0 in ( 64, 32 ) if not ( "stripes" in kind and n0 == 32 ) ]
    cpu = _cpu_solves( cases )
    for ( q, n0 ), c in zip( cases, cpu ):
        kind, target = targets[ q ]
        p = OtProblem( SumOfDiracsOnAGrid( ( 64, 64 ), weights = m ), target() ).solve( Iterative( tol = 1e-12, tuning = Tuning( grid_n0 = n0 ) ) )
        st, cs = p.stats, c[ "stats" ]
        print( f"  { kind }, n0 = { n0 }: card { st[ 'nb_iter' ] } its / { st[ 'nb_diag' ] } diagrams / { st[ 'lin_nb_iter' ] } CG, "
               f"cpu { cs[ 'nb_iter' ] } / { cs[ 'nb_diag' ] } / { cs[ 'lin_nb_iter' ] } ( { cs[ 'grid_solver' ] } )" )
        assert p.converged and cs[ "status" ] == "converged", ( kind, n0, st[ "status" ], cs[ "status" ] )
        w = _host( p.weights )
        assert numpy.allclose( w - w[ 0 ], c[ "weights" ] - c[ "weights" ][ 0 ], atol = 1e-9 ), ( kind, n0, numpy.abs( ( w - w[ 0 ] ) - ( c[ "weights" ] - c[ "weights" ][ 0 ] ) ).max() )
        assert numpy.allclose( _host( p.cell_masses ), c[ "masses" ], atol = 1e-11 ), ( kind, n0 )
        assert numpy.allclose( _host( p.barycenters ), c[ "bary" ], atol = 1e-8 ), ( kind, n0, numpy.abs( _host( p.barycenters ) - c[ "bary" ] ).max() )
        # ( gaussians: the CPU's moments are a quadrature, the card's closed forms )
        assert numpy.isclose( float( _host( p.cost ) ), c[ "cost" ], rtol = 1e-6 if kind == "gaussians" else 1e-9 ), ( kind, n0, float( _host( p.cost ) ), c[ "cost" ] )

if test( "the_card_grid_solver_starts_from_given_weights" ):
    import numpy
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem
    _on_card()
    m = _grid_masses( 64 )
    target = _targets()[ 1 ][ 1 ]
    p = OtProblem( SumOfDiracsOnAGrid( ( 64, 64 ), weights = m ), target() ).solve( Iterative( tol = 1e-12 ) )
    q = OtProblem( SumOfDiracsOnAGrid( ( 64, 64 ), weights = m ), target() ).solve( Iterative( tol = 1e-12, weights0 = _host( p.weights ) ) )
    assert q.converged and q.stats[ "nb_iter" ] <= 1 and q.stats[ "nb_continuation_steps" ] == 1, q.stats
    assert numpy.allclose( _host( q.weights ), _host( p.weights ), atol = 1e-10 )

if test( "the_card_grid_solver_on_a_large_grid" ):
    import numpy
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem, Tuning
    _on_card()
    # coarse to fine from 64 up to 512, every density; the masses reached ( relative )
    m = _grid_masses( 512 )
    for kind, target in _targets():
        if "stripes" in kind:
            continue
        p = OtProblem( SumOfDiracsOnAGrid( ( 512, 512 ), weights = m ), target() ).solve( Iterative( tol = 0.0, tuning = Tuning( mass_rtol = 1e-9 ) ) )
        assert p.converged and p.stats[ "grid_solver" ] == f"dedicated card ( { kind.replace( " stripes", "" ) } )", ( kind, p.stats )
        cm = _host( p.cell_masses )
        assert numpy.abs( cm / cm.sum() - m ).max() <= 1e-8 * m.max(), ( kind, numpy.abs( cm / cm.sum() - m ).max() )

if test( "what_the_card_grid_solver_does_not_take_goes_to_the_positions" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem
    _on_card()
    # not N x N with N a power of two; a target that is not on the box of the grid
    for g, target in ( ( SumOfDiracsOnAGrid( ( 24, 24 ) ), Box( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ),
                       ( SumOfDiracsOnAGrid( ( 32, 32 ), frame = [ 0.5, 0.5 ] ), Box( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ) ):
        p = OtProblem( g, target ).solve( Iterative( tol = 1e-10 ) )
        assert p.converged and "grid_solver" not in p.stats, p.stats
