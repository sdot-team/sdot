from .distributions.distribution_methods import generate_batch_version_of, generate_1d_version_of
from .distributions.BatchOfDistributions import BatchOfDistributions

from .distributions.PiecewiseAffineFunction1d import PiecewiseAffineFunction1d
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs
from .distributions.Distribution import Distribution

from .driver import driver as _driver
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    class SumOfWeightedDiracs1d( SumOfWeightedDiracs ):
        positions : Any
        def __init__( self, positions = None, weights = None ): ...

    class BatchOfSumOfWeightedDiracs( BatchOfDistributions ):
        nb_diracs : int
        positions : Any
        weights   : Any
        def __init__( self, positions = None, weights = None ): ...

    class BatchOfSumOfWeightedDiracs1d( BatchOfSumOfWeightedDiracs ):
        def __init__( self, positions = None, weights = None ): ...

    class BatchOfPiecewiseAffineFunction1d( BatchOfDistributions ):
        nb_points : int
        xs        : Any
        ys        : Any
        x0        : float
        x1        : float
        def __init__( self, xs = None, ys = None, x0: float = 0, x1: float = 1 ): ...

else:
    SumOfWeightedDiracs1d            = generate_1d_version_of( SumOfWeightedDiracs )
    BatchOfSumOfWeightedDiracs       = generate_batch_version_of( SumOfWeightedDiracs )
    BatchOfSumOfWeightedDiracs1d     = generate_batch_version_of( SumOfWeightedDiracs1d )
    BatchOfPiecewiseAffineFunction1d = generate_batch_version_of( PiecewiseAffineFunction1d )

from .BatchOf1dOtPlans import BatchOf1dOtPlans as BatchOf1dOtPlans
from .BatchOfOtPlans import BatchOfOtPlans as BatchOfOtPlans
from .OtPlan1d import OtPlan1d as OtPlan1d
from .OtPlan import OtPlan as OtPlan

driver = _driver

__all__ = [
    "BatchOfPiecewiseAffineFunction1d",
    "BatchOfSumOfWeightedDiracs1d",
    "BatchOfSumOfWeightedDiracs",
    "PiecewiseAffineFunction1d",
    "SumOfWeightedDiracs1d",
    "BatchOfDistributions",
    "SumOfWeightedDiracs",
    "BatchOf1dOtPlans",
    "BatchOfOtPlans",
    "Distribution",
    "barycenters",
    "distances",
    "distance",
    "OtPlan1d",
    "OtPlan",
    "driver",
    "plan",
]

def plan( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions, _check_1d = True ) -> OtPlan | OtPlan1d | BatchOfOtPlans | BatchOf1dOtPlans:
    # ensure batch
    if isinstance( f, Distribution ) and isinstance( g, Distribution ):
        return plan( f.batch_version( 1 ), g.batch_version( 1 ) ).unbatch()
    if isinstance( f, Distribution ):
        return plan( f.batch_version( g.batch_size ), g )
    if isinstance( g, Distribution ):
        return plan( f, g.batch_version( f.batch_size ) )

    # always unidimensional
    if _check_1d and f.always_1d:
        return plan( f, g, _check_1d = False ).unidimensionnal_version()

    # ensure `f` is a BatchOfSumOfWeightedDiracs, even if it means swapping `f` and `g`
    if not isinstance( f, BatchOfSumOfWeightedDiracs ):
        if isinstance( g, BatchOfSumOfWeightedDiracs ):
            return plan( g, f )
        raise RuntimeError( "TODO: handle cases where f and g are both _not_ SumOfWeightedDiracs" )

    # dispatch on type of g
    try:
        method = driver.map_of_plan_methods[ type( g ).__name__ ]
    except KeyError:
        raise RuntimeError( f"TODO: SumOfWeightedDiracs -> { type( g ) }" )

    return method( f, g )


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
    BatchOfSumOfWeightedDiracs1d()
    PiecewiseAffineFunction1d()
    SumOfWeightedDiracs1d()
    SumOfWeightedDiracs()
