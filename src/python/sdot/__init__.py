# base distributions (1d and batch version are generated after)
# from .distributions.PiecewiseConstantImage import PiecewiseConstantImage as PiecewiseConstantImage
from .distributions.PiecewiseConstantGrid import PiecewiseConstantGrid as PiecewiseConstantGrid
from .distributions.BatchOfDistributions import BatchOfDistributions as BatchOfDistributions
from .distributions.PiecewiseAffineGrid import PiecewiseAffineGrid as PiecewiseAffineGrid
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs as SumOfWeightedDiracs
from .distributions.PolynomialGrid import PolynomialGrid as PolynomialGrid
from .distributions.Distribution import Distribution as Distribution

from .distributions.helpers.distribution_methods import variants_of


# Ot plans
from .BatchOf1dOtPlans import BatchOf1dOtPlans as BatchOf1dOtPlans
from .BatchOfOtPlans import BatchOfOtPlans as BatchOfOtPlans
from .OtPlan1d import OtPlan1d as OtPlan1d
from .OtPlan import OtPlan as OtPlan

from .Polynomial import Polynomial as Polynomial
from .Cell import Cell as Cell
from .Bsp import Bsp as Bsp

from .ot_plan import ot_plan as ot_plan, distance as distance, distances as distances, barycenters as barycenters, make_apply_fn as make_apply_fn

# solvers
from .OtSolver import OtSolver as OtSolver

# hardware/framework interfaces
from .driver import driver as driver


# variants
SumOfWeightedDiracs1d, BatchOfSumOfWeightedDiracs, BatchOfSumOfWeightedDiracs1d = variants_of( SumOfWeightedDiracs )
PiecewiseAffineGrid1d, BatchOfPiecewiseAffineGrid, BatchOfPiecewiseAffineGrid1d = variants_of( PiecewiseAffineGrid )

