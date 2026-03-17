from ..BatchOfOtPlans import BatchOfOtPlans
from ..bindings import sdot_bindings_cpu
import torch

class PyTorchDriver:
    """

    """
    def __init__( self, dtype = torch.float32, device = None ):
        if type( dtype ) == str:
            dtype = getattr( torch, dtype )
            assert isinstance( dtype, torch.dtype )

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

    def repeat( self, tensor, shape ):
        return tensor.repeat( shape )

    def batch_of_ot_plan_for_Piecewise1dAffineFunctions( self, f, g ) -> BatchOfOtPlans:
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
                    grad_distance.contiguous(), grad_barycenters.contiguous(), dirac_xs, dirac_ws, point_xs, point_ys, barycenters,
                    grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys
                )

                return grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys

        distances, barycenters = SDOTFunction.apply( f._nd_positions(), f.weights, g.xs, g.ys )
        return BatchOfOtPlans( distances, barycenters )

    def optimize_using_lbfgs( self, loss, params, on_iter = None ):
        """ small helper to optimize `loss` wrt `params` using lbfgs """
        lbfgs = torch.optim.LBFGS( [ params ], history_size = 15, max_iter = 10 ) # , line_search_fn = "strong_wolfe"

        def closure():
            lbfgs.zero_grad()
            objective = loss( params )
            objective.backward()
            return objective

        old_params = params.clone().detach()
        tol_param = 1e-7  # Stabilité de la solution
        tol_grad  = 1e-7  # Stabilité du gradient (proche du minimum)
        for i in range( 50 ):
            lbfgs.step( closure )

            if on_iter:
                on_iter( params, i )

            # --- CRITÈRES DE CONVERGENCE ---
            with torch.no_grad():
                grad_norm = torch.norm( params.grad )
                param_diff = torch.norm( params - old_params )

                print(f"Itération {i:02d} | Grad Norm: {grad_norm:.2e} | Param Diff: {param_diff:.2e}")

                # Test de sortie
                if grad_norm < tol_grad:
                    # print( "=> Convergence : Le gradient est quasiment nul." )
                    break
                if param_diff < tol_param:
                    # print( "=> Convergence : La solution est stabilisée." )
                    break

                old_params.copy_( params )

    def optimize_using_sgd( self, loss, params ):
        optimizer = torch.optim.SGD( [ params ] )
        for _ in range( 50 ):
            l = loss( params )
            print( l )
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
