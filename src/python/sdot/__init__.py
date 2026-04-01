# base distributions (1d and batch version are generated after)
# from .distributions.PiecewiseConstantImage import PiecewiseConstantImage as PiecewiseConstantImage
from .distributions.BatchOfDistributions import BatchOfDistributions as BatchOfDistributions
from .distributions.PiecewiseAffineGrid import PiecewiseAffineGrid as PiecewiseAffineGrid
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs as SumOfWeightedDiracs
from .distributions.Distribution import Distribution as Distribution

from .distributions.helpers.distribution_methods import variants_of


# Ot plans
from .BatchOf1dOtPlans import BatchOf1dOtPlans as BatchOf1dOtPlans
from .BatchOfOtPlans import BatchOfOtPlans as BatchOfOtPlans
from .OtPlan1d import OtPlan1d as OtPlan1d
from .OtPlan import OtPlan as OtPlan
from .Cell import Cell as Cell
from .Bsp import Bsp as Bsp

from .plan import plan as plan, distance as distance, distances as distances, barycenters as barycenters


# hardware/framework interfaces
from .driver import driver as driver


# variants
SumOfWeightedDiracs1d, BatchOfSumOfWeightedDiracs, BatchOfSumOfWeightedDiracs1d = variants_of( SumOfWeightedDiracs )
PiecewiseAffineGrid1d, BatchOfPiecewiseAffineGrid, BatchOfPiecewiseAffineGrid1d = variants_of( PiecewiseAffineGrid )

# # generated variants
# if TYPE_CHECKING:
#     class SumOfWeightedDiracs1d( SumOfWeightedDiracs ):
#         positions : Any
#         def __init__( self, positions = None, weights = None ): ...

#     class BatchOfSumOfWeightedDiracs( BatchOfDistributions ):
#         nb_diracs : int
#         positions : Any
#         weights   : Any
#         def __init__( self, positions = None, weights = None ): ...

#     class BatchOfPiecewiseConstantImage( BatchOfDistributions ):
#         values  : Any
#         corners : Any
#         def __init__( self, values = None, corners = None ): ...

#     class BatchOfSumOfWeightedDiracs1d( BatchOfSumOfWeightedDiracs ):
#         def __init__( self, positions = None, weights = None ): ...

#     class BatchOfPiecewiseAffineGrid1d( BatchOfDistributions ):
#         nb_points : int
#         xs        : Any
#         ys        : Any
#         x0        : float
#         x1        : float
#         def __init__( self, xs = None, ys = None, x0: float = 0, x1: float = 1 ): ...

# else:
#     SumOfWeightedDiracs1d            = generate_1d_version_of( SumOfWeightedDiracs )

#     BatchOfSumOfWeightedDiracs       = generate_batch_version_of( SumOfWeightedDiracs )
#     BatchOfSumOfWeightedDiracs1d     = generate_batch_version_of( SumOfWeightedDiracs1d )
#     BatchOfPiecewiseConstantImage    = generate_batch_version_of( PiecewiseConstantImage )
#     BatchOfPiecewiseAffineGrid1d = generate_batch_version_of( PiecewiseAffineGrid1d )
