from ..BatchOfOtPlans import BatchOfOtPlans
from ..bindings import sdot_bindings_cpu
import torch
import sdot

class PyTorchDriver:
    """

    """
    def __init__( self, dtype = torch.float32, device = None ):
        self.device = device or torch.get_default_device()
        self.dtype = dtype

    def t3( self, tensor ):
        """ make a rank 3 tensor """
        return self.tn( tensor, 3 )

    def t2( self, tensor ):
        """ make a rank 2 tensor """
        return self.tn( tensor, 2 )

    def t1( self, tensor ):
        """ make a rank 1 tensor """
        return self.tn( tensor, 1 )

    def t0( self, tensor ):
        """ make a rank 0 tensor """
        return self.tn( tensor, 0 )

    def tn( self, tensor, ndim ):
        """ make a rank ndim tensor """
        if tensor is None:
            return tensor
        res = torch.as_tensor( tensor, dtype = self.dtype, device = self.device )
        assert res.ndim == ndim
        return res

    def ones( self, shape ):
        return torch.ones( shape, dtype = self.dtype, device = self.device )

    def linspace( self, a, b, n ):
        return torch.linspace( a, b, n, dtype = self.dtype, device = self.device )

    def empty( self, shape ):
        return torch.ones( shape, dtype = self.dtype, device = self.device )

    def batch_of_ot_plan_for_Piecewise1dAffineFunctions( self, f: sdot.BatchOfSumOfWeightedDiracs, g: sdot.BatchOfPiecewise1dAffineFunctions ) -> BatchOfOtPlans:
        class SDOTFunction( torch.autograd.Function ):
            @staticmethod
            def forward( ctx, dirac_xs, dirac_ws, point_xs, point_ys ) -> tuple[ torch.tensor, torch.tensor ]:
                distances = self.empty( dirac_xs.shape[ 0 : 1 ] )
                barycenters = self.empty( dirac_xs.shape )

                sdot_bindings_cpu.ot_plan_to_piecewise_affine_1d( dirac_xs, dirac_ws, point_xs, point_ys, distances, barycenters )

                ctx.save_for_backward( dirac_xs, dirac_ws, point_xs, point_ys, barycenters )
                return distances, barycenters

            @staticmethod
            def backward( ctx, grad_distance, grad_barycenters ):
                dirac_xs, dirac_ws, point_xs, point_ys, barycenters = ctx.saved_tensors
                grad_dirac_xs = self.empty( dirac_xs.shape )
                grad_dirac_ws = self.empty( dirac_ws.shape )
                grad_point_xs = self.empty( point_xs.shape )
                grad_point_ys = self.empty( point_ys.shape )

                sdot_bindings_cpu.backward_ot_plan_to_piecewise_affine_1d(
                    grad_distance, grad_barycenters, dirac_xs, dirac_ws, point_xs, point_ys, barycenters,
                    grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys
                )

                return grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys

        distances, barycenters = SDOTFunction.apply( f._nd_positions(), f.weights, g.xs, g.ys )
        return BatchOfOtPlans( distances, barycenters )
