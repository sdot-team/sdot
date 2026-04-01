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

from .ot_plan import ot_plan as ot_plan, distance as distance, distances as distances, barycenters as barycenters


# hardware/framework interfaces
from .driver import driver as driver


# variants
SumOfWeightedDiracs1d, BatchOfSumOfWeightedDiracs, BatchOfSumOfWeightedDiracs1d = variants_of( SumOfWeightedDiracs )
PiecewiseAffineGrid1d, BatchOfPiecewiseAffineGrid, BatchOfPiecewiseAffineGrid1d = variants_of( PiecewiseAffineGrid )

