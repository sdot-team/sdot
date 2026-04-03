from ..distributions.BatchOfDistributions import BatchOfDistributions, unflatten_args
from ..BatchOfOtPlans import BatchOfOtPlans
from ..driver import driver
import numpy as np
import torch

class PyTorchDriver:
    """

    """
    def __init__( self, normalized_dtype: str, normalized_device: str ):
        self.device = PyTorchDriver.find_device( normalized_device )
        self.dtype = PyTorchDriver.find_dtype( normalized_dtype )

    @property
    def name( self ) -> str:
        return "torch"

    @staticmethod
    def default_normalized_device_for( user_normalized_dtype ):
        if torch.cuda.is_available():
            return "cuda:0"
        # Metal (MPS) only supports FP32
        if torch.backends.mps.is_available() and user_normalized_dtype == "FP32":
            return "metal"
        return "cpu"


    @staticmethod
    def find_device( normalized_device: str ):
        """ find the torch device from a normalized name like cpu, cuda:1, metal """
        if normalized_device.startswith( "cpu" ):
            return torch.device( "cpu" )
        if normalized_device.startswith( "cuda" ):
            # idx = normalized_device.split( ":" )[ 1 ] if ":" in normalized_device else "0"
            return torch.device( "cuda" ) # :{ idx }
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

    def tn( self, tensor, ndim, name = None ):
        """ make a rank ndim tensor """
        if tensor is None:
            return tensor
        res = torch.as_tensor( tensor, dtype = self.dtype, device = self.device )
        if ndim is not None and res.ndim != ndim:
            if name is not None:
                raise IndexError( f"expecting for field '{ name }' a { ndim }d tensor, but { res.ndim }d was provided." )
            raise IndexError( f"expecting a { ndim }d tensor, but { res.ndim }d was provided." )
        return res

    def zeros( self, shape ):
        return torch.zeros( shape, dtype = self.dtype, device = self.device )

    def ones( self, shape ):
        return torch.ones( shape, dtype = self.dtype, device = self.device )

    def linspace( self, a, b, n ):
        return torch.linspace( a, b, n, dtype = self.dtype, device = self.device )

    def empty( self, shape ):
        return torch.zeros( shape, dtype = self.dtype, device = self.device )

    def expand_dims( self, tensor, index ):
        return tensor.unsqueeze( index )

    def repeat( self, tensor, shape ):
        return tensor.repeat( shape )

    def hstack( self, lst ):
        return torch.hstack( lst )

    def to_numpy( self, t ):
        return np.array( t )
        # if isinstance( t, list ):
        # return t.to_numpy()

    def plan( self, bindings, f: BatchOfDistributions, g: BatchOfDistributions ):
        class SDOTFunction( torch.autograd.Function ):
            @staticmethod
            def forward( ctx, dirac_xs, *args ):
                # get constants
                batch_size = dirac_xs.shape[ 0 ]
                nb_diracs = dirac_xs.shape[ 1 ]
                dim = dirac_xs.shape[ 2 ]

                # room for the outputs
                barycenters = self.empty( [ batch_size, nb_diracs, dim ] )
                potentials = self.empty( [ batch_size, nb_diracs ] )
                distances = self.empty( [ batch_size ] )
                cuts = self.empty( [ batch_size, nb_diracs, 2 ] )

                # arguments as expected by the binding
                binding_inputs = unflatten_args( f, g, [ dirac_xs ] + list( args ) )

                # call the C++ procedure
                bindings.forward( *binding_inputs, distances, barycenters, potentials, cuts )

                ctx.save_for_backward( dirac_xs, *args, distances, barycenters, potentials, cuts )

                return distances, barycenters, potentials, cuts

            @staticmethod
            def backward( ctx, grad_distance, grad_barycenters, grad_potentials, grad_cuts ):
                # get room for the output gradients
                flat_grad_outputs = [ self.empty( arg.shape ) for arg in ctx.saved_tensors[ :-4 ] ]

                # arguments as expected by the binding
                binding_grad_outputs = unflatten_args( f, g, flat_grad_outputs )
                binding_inputs = unflatten_args( f, g, ctx.saved_tensors[ :-4 ] )

                bindings.backward( *binding_inputs, *ctx.saved_tensors[ -4: ], grad_distance, grad_barycenters, grad_potentials, grad_cuts, *binding_grad_outputs )

                return tuple( flat_grad_outputs )

        input_tensors = f.flat_tensor_list() + g.flat_tensor_list()
        outputs = SDOTFunction.apply( *input_tensors )
        assert isinstance( outputs, tuple )
        return BatchOfOtPlans( *outputs )

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
