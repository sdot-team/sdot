from .distributions.BatchOfDistributions import BatchOfDistributions
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs
from .distributions.Distribution import Distribution
from .BatchOf1dOtPlans import BatchOf1dOtPlans
from .BatchOfOtPlans import BatchOfOtPlans
from .OtPlan1d import OtPlan1d
from .OtPlan import OtPlan
from .driver import driver


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
    if not isinstance( f, SumOfWeightedDiracs.batch_class() ):
        if isinstance( g, SumOfWeightedDiracs.batch_class() ):
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
