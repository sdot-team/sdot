from .distributions.BatchOfPiecewise1dAffineFunctions import BatchOfPiecewise1dAffineFunctions
from .distributions.BatchOfSumOfWeighted1dDiracs import BatchOfSumOfWeighted1dDiracs
from .distributions.BatchOfSumOfWeightedDiracs import BatchOfSumOfWeightedDiracs
from .distributions.BatchOfDistributions import BatchOfDistributions

from .distributions.Piecewise1dAffineFunction import Piecewise1dAffineFunction
from .distributions.SumOf1dWeightedDiracs import SumOf1dWeightedDiracs
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs
from .distributions.Distribution import Distribution

from .BatchOf1dOtPlans import BatchOf1dOtPlans
from .BatchOfOtPlans import BatchOfOtPlans
from .OtPlan1d import OtPlan1d
from .OtPlan import OtPlan

from .driver import driver, use_jax, use_pytorch


def plan( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions, _check_1d = True ) -> OtPlan | OtPlan1d | BatchOfOtPlans | BatchOf1dOtPlans:
    # ensure batch
    if isinstance( f, Distribution ) and isinstance( g, Distribution ):
        return plan( f.batch_version(), g.batch_version() ).unbatch()
    if isinstance( f, Distribution ):
        return plan( f.batch_version( g.batch_size ), g )
    if isinstance( g, Distribution ):
        return plan( f, g.batch_version( f.batch_size ) )

    # always unidimensionnal
    if _check_1d and f.always_1d:
        return plan( f, g, _check_1d = False ).unidimensionnal_version()

    # ensure `f` is a BatchOfSumOfWeightedDiracs, even if it means swaping `f` and `g`
    if not isinstance( f, BatchOfSumOfWeightedDiracs ):
        if isinstance( g, BatchOfSumOfWeightedDiracs ):
            return plan( g, f )
        raise RuntimeError( "TODO: handle cases where f and g are both _not_ SumOfWeightedDiracs" )

    # case BatchOfPiecewise1dAffineFunctions
    if isinstance( g, BatchOfPiecewise1dAffineFunctions ):
        assert f.dim == 1
        return driver.batch_of_ot_plan_for_Piecewise1dAffineFunctions( f, g )

    # unhandled case
    raise RuntimeError( f"TODO: SumOfWeightedDiracs -> { type( g ) }" )


def distances( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    return plan( f, g ).distances

def distance( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    d = distances( f, g )
    if d.ndim == 1:
        if d.shape[ 0 ] != 1:
            raise RuntimeError( "sdot.distance works only for batch_size = 1" )
        return d[ 0 ]
    assert d.ndim == 0
    return d.item()


def barycenters( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    return plan( f, g ).barycenters


def __():
    """ fake function """
    BatchOfSumOfWeighted1dDiracs()
    Piecewise1dAffineFunction()
    SumOf1dWeightedDiracs()
    SumOfWeightedDiracs()
    # helpers.solve_bfgs
    # PiecewiseAffineFunction()
    # SumOfWeightedDiracs()
    # driver.ones( [] )
