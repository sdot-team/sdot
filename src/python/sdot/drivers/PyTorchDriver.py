# import sdot
import torch

map_of_plan_methods = {}


class PyTorchDriver:
    """

    """
    def __init__( self, normalized_dtype: str, normalized_device: str ):
        self.device = PyTorchDriver.find_device( normalized_device )
        self.dtype = PyTorchDriver.find_dtype( normalized_dtype )

        self.map_of_plan_methods = map_of_plan_methods

    @staticmethod
    def default_dtype( normalized_device: str | None ):
        """ normalized_device can be "gpu:...", "cpu", "metal" or None
            return appropriate type (metal => float, cuda => dep)
        """
        if normalized_device == "metal":
            return "FP32"
        return "FP64"

    @staticmethod
    def default_device( normalized_dtype ):
        """
        normalized_dtype can be None, "FP32" or "FP64"
        """
        if torch.cuda.is_available():
            return "cuda:0"
        # Metal (MPS) only supports FP32
        if torch.backends.mps.is_available() and normalized_dtype != "FP64":
            return "metal"
        return "cpu"

    @staticmethod
    def find_device( normalized_device: str ):
        """ find the torch device from a normalized name like cpu, cuda:1, metal """
        if normalized_device.startswith( "cpu" ):
            return torch.device( "cpu" )
        if normalized_device.startswith( "cuda" ):
            idx = normalized_device.split( ":" )[ 1 ] if ":" in normalized_device else "0"
            return torch.device( f"cuda:{ idx }" )
        if normalized_device.startswith( "metal" ):
            return torch.device( "mps" )
        raise RuntimeError( f"Unknown type { normalized_device }" )

    @staticmethod
    def find_dtype( normalized_dtype: str ):
        if normalized_dtype == "FP32":
            return torch.float32
        if normalized_dtype == "FP64":
            return torch.float64
        raise RuntimeError( f"Unknown type { normalized_dtype }" )

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
        return torch.zeros( shape, dtype = self.dtype, device = self.device )

    def repeat( self, tensor, shape ):
        return tensor.repeat( shape )

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

                print(f"Itération {i:02d} | Grad Norm: {grad_norm.item():.2e} | Param Diff: {param_diff.item():.2e}")

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
        for _ in range( 20 ):
            l = loss( params )
            print( l )
            optimizer.zero_grad()
            l.backward()
            optimizer.step()


from .pytorch_functions import ot_plan_for_Piecewise1dAffineFunctions
