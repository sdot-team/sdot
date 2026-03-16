from .PiecewiseAffineFunction import PiecewiseAffineFunction
from .SumOfWeightedDiracs import SumOfWeightedDiracs
from .Distribution import Distribution
from .OtPlan import OtPlan
from .driver import driver

from sdot.bindings import sdot_bindings_cpu


def plan( f: Distribution, g: Distribution ) -> OtPlan:
    if type( g ) == SumOfWeightedDiracs:
        return plan( g, f )
    if not isinstance( f, SumOfWeightedDiracs ):
        raise RuntimeError( "TODO: handle cases where f and g are not SumOfWeightedDiracs" )

    barycenters = driver.empty( [ f.batch_size, f.nb_diracs ] )
    distances = driver.empty( [ f.batch_size ] )

    if isinstance( g, PiecewiseAffineFunction ):
        sdot_bindings_cpu.forward_impl( f.positions, f.weights, g.xs, g.ys, distances, barycenters )
        return OtPlan( distances, barycenters )

    raise RuntimeError( f"TODO: SumOfWeightedDiracs -> { type( g ) }" )


def distances( f: Distribution, g: Distribution ):
    return plan( f, g ).distances

def distance( f: Distribution, g: Distribution ):
    d = distances( f, g )
    if d.shape[ 0 ] != 1:
        raise RuntimeError( "sdot.distance works only for batch_size = 1" )
    return d[ 0, : ]

def __():
    """ fake function """
    helpers.solve_bfgs
    # PiecewiseAffineFunction()
    # SumOfWeightedDiracs()
    # driver.ones( [] )
