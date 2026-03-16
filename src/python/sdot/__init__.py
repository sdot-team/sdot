from .distributions.BatchOfPiecewise1dAffineFunctions import BatchOfPiecewise1dAffineFunctions
from .distributions.BatchOfSumOf1dWeightedDiracs import BatchOfSumOf1dWeightedDiracs
from .distributions.BatchOfSumOfWeightedDiracs import BatchOfSumOfWeightedDiracs
from .distributions.BatchOfDistributions import BatchOfDistributions

from .distributions.Piecewise1dAffineFunction import Piecewise1dAffineFunction
from .distributions.SumOf1dWeightedDiracs import SumOf1dWeightedDiracs
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs
from .distributions.Distribution import Distribution

from sdot.bindings import sdot_bindings_cpu

from .BatchOfOtPlan import BatchOfOtPlan
from .OtPlan import OtPlan
from .driver import driver


def plan( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions, _output_cb = lambda x: x, _check_1d = True ) -> OtPlan:
    # ensure batch
    if isinstance( f, Distribution ) and isinstance( g, Distribution ):
        return plan( f.batch_version(), g.batch_version(), _output_cb = lambda x: x.unbatch() )
    if isinstance( f, Distribution ):
        return plan( f.batch_version(), g )
    if isinstance( g, Distribution ):
        return plan( f, g.batch_version() )

    # always unidimensionnal
    if _check_1d and f.always_1d:
        return plan( f, g, _output_cb = lambda x: _output_cb( x ).unidimensionnal_version(), _check_1d = False )

    # ensure `f` is a BatchOfSumOfWeightedDiracs, even if it means swaping `f` and `g`
    if not isinstance( f, BatchOfSumOfWeightedDiracs ):
        if isinstance( g, BatchOfSumOfWeightedDiracs ):
            return plan( g, f )
        raise RuntimeError( "TODO: handle cases where f and g are both _not_ SumOfWeightedDiracs" )

    # case BatchOfPiecewise1dAffineFunctions
    if isinstance( g, BatchOfPiecewise1dAffineFunctions ):
        barycenters = driver.empty( [ f.batch_size, f.nb_diracs, 1 ] )
        distances = driver.empty( [ f.batch_size ] )
        assert f.dim == 1

        sdot_bindings_cpu.ot_plan_to_piecewise_affine_1d(
            f._nd_positions()[ :, :, 0 ], f.weights, g.xs, g.ys, distances, barycenters
        )

        return _output_cb( BatchOfOtPlan( distances, barycenters ) )

    # unhandled case
    raise RuntimeError( f"TODO: SumOfWeightedDiracs -> { type( g ) }" )


def distances( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    return plan( f, g ).distances

def distance( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    d = distances( f, g )
    if d.shape[ 0 ] != 1:
        raise RuntimeError( "sdot.distance works only for batch_size = 1" )
    return d[ 0 ]

def __():
    """ fake function """
    BatchOfSumOf1dWeightedDiracs()
    Piecewise1dAffineFunction()
    SumOf1dWeightedDiracs()
    SumOfWeightedDiracs()
    # helpers.solve_bfgs
    # PiecewiseAffineFunction()
    # SumOfWeightedDiracs()
    # driver.ones( [] )
