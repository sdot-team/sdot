from ...BatchOfOtPlans import BatchOfOtPlans
from ...bindings import sdot_bindings_cpu
from ...driver import driver

from ..PyTorchDriver import map_of_plan_methods

import torch


class SDOTFunction( torch.autograd.Function ):
    @staticmethod
    def forward( ctx, dirac_xs, dirac_ws, point_xs, point_ys ) -> tuple[ torch.tensor, torch.tensor, torch.tensor, torch.tensor ]:
        dirac_xs = dirac_xs.contiguous()
        dirac_ws = dirac_ws.contiguous()
        point_xs = point_xs.contiguous()
        point_ys = point_ys.contiguous()

        batch_size = dirac_xs.shape[ 0 ]
        nb_diracs = dirac_xs.shape[ 1 ]
        dim = dirac_xs.shape[ 2 ]

        barycenters = driver.empty( [ batch_size, nb_diracs, dim ] )
        potentials = driver.empty( [ batch_size, nb_diracs ] )
        distances = driver.empty( [ batch_size ] )
        cuts = driver.empty( [ batch_size, nb_diracs, 2 ] )

        getattr( sdot_bindings_cpu, "ot_plan_to_piecewise_affine_1d_" + driver.normalized_dtype )(
            dirac_xs, dirac_ws, point_xs, point_ys,
            distances, barycenters, potentials, cuts
        )

        ctx.save_for_backward( dirac_xs, dirac_ws, point_xs, point_ys, barycenters, potentials, cuts )
        return distances, barycenters, potentials, cuts

    @staticmethod
    def backward( ctx, grad_distance, grad_barycenters, grad_potentials, grad_cuts ):
        dirac_xs, dirac_ws, point_xs, point_ys, barycenters, potentials, cuts = ctx.saved_tensors
        grad_dirac_xs = driver.empty( dirac_xs.shape )
        grad_dirac_ws = driver.empty( dirac_ws.shape )
        grad_point_xs = driver.empty( point_xs.shape )
        grad_point_ys = driver.empty( point_ys.shape )

        getattr( sdot_bindings_cpu, "backward_ot_plan_to_piecewise_affine_1d_" + driver.normalized_dtype )(
            grad_distance.contiguous(),
            grad_barycenters.contiguous(),
            dirac_xs, dirac_ws, point_xs, point_ys,
            barycenters, potentials, cuts,
            grad_dirac_xs.contiguous(), 
            grad_dirac_ws.contiguous(), 
            grad_point_xs.contiguous(), 
            grad_point_ys.contiguous()
        )

        return grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys


def ot_plan_for_Piecewise1dAffineFunctions( f, g ) -> BatchOfOtPlans:
    """ : sdot.BatchOfPiecewise1dAffineFunctions """
    distances, barycenters, potentials, cuts = SDOTFunction.apply( f._nd_positions(), f.weights, g.xs, g.ys )
    return BatchOfOtPlans( distances, barycenters, potentials, cuts )


map_of_plan_methods[ "BatchOfPiecewise1dAffineFunctions" ] = ot_plan_for_Piecewise1dAffineFunctions
