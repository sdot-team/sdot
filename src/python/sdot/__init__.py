# base distributions (1d and batch version are generated after)
from .distributions.PiecewiseAffineFunction1d import PiecewiseAffineFunction1d as PiecewiseAffineFunction1d
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs as SumOfWeightedDiracs
from .distributions.Distribution import Distribution as Distribution


# Ot plans
from .BatchOf1dOtPlans import BatchOf1dOtPlans as BatchOf1dOtPlans
from .BatchOfOtPlans import BatchOfOtPlans as BatchOfOtPlans
from .OtPlan1d import OtPlan1d as OtPlan1d
from .OtPlan import OtPlan as OtPlan
from .Bsp import Bsp as Bsp

from .plan import plan as plan, distance as distance, distances as distances, barycenters as barycenters


# hardware/framework interfaces
from .driver import driver as driver


# tools to generate variants for the distributions
from .distributions.distribution_methods import generate_batch_version_of as generate_batch_version_of, generate_1d_version_of as generate_1d_version_of
from .distributions.BatchOfDistributions import BatchOfDistributions as BatchOfDistributions
from typing import TYPE_CHECKING, Any


# generated variants
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


driver = driver


