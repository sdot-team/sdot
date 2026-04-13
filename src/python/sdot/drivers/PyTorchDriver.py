from ..distributions.helpers.distribution_methods import unflatten_args, flat_tensor_list, to_tensor_list
from ..distributions.BatchOfDistributions import BatchOfDistributions
from ..BatchOfOtPlans import BatchOfOtPlans
# from ..driver import driver
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

    @property
    def array_type( self ):
        return torch.Tensor

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

    def tn( self, tensor, ndim = None, name = None ):
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

    def stack( self, tensors, axis ):
        return torch.stack( tensors, dim=axis )

    def linalg_solve( self, A, b ):
        return torch.linalg.solve( A, b )

    def moveaxis( self, tensor, source, destination ):
        return torch.moveaxis( tensor, source, destination )

    def hstack( self, lst ):
        return torch.hstack( lst )

    def to_numpy( self, t ):
        return np.array( t )
        # if isinstance( t, list ):
        # return t.to_numpy()

    def forward( self, forward_func, backward_func, out_shapes, *inputs ):
        """ Generic differentiable wrapper with explicit backward.
            out_shapes    : list of shape tuples, e.g. [ (batch,), (batch, n, dim) ]
            forward_func ( *np_outputs, *np_inputs    ) -> None  (fills np_outputs in-place)
            backward_func( *np_grad_inputs, *np_outputs,
                           *np_grad_outputs, *np_inputs ) -> None  (fills np_grad_inputs in-place)
            Mutable (output) arrays come first, matching the nanobind convention.
        """
        np_dtype      = { torch.float32: np.float32, torch.float64: np.float64 }[ self.dtype ]
        input_tensors = to_tensor_list( inputs )
        n_out         = len( out_shapes )

        outer_self = self

        class Func( torch.autograd.Function ):
            @staticmethod
            def forward( ctx, *t_inputs ):
                np_inputs  = [ t.cpu().detach().numpy() for t in t_inputs ]
                np_outputs = [ np.empty( shape, dtype = np_dtype ) for shape in out_shapes ]
                forward_func( *np_outputs, *np_inputs )

                t_outputs = [ torch.as_tensor( o, dtype = outer_self.dtype, device = outer_self.device ) for o in np_outputs ]
                ctx.save_for_backward( *t_inputs, *t_outputs )
                return tuple( t_outputs )

            @staticmethod
            def backward( ctx, *grad_outputs ):
                saved      = ctx.saved_tensors
                t_inputs   = saved[ :-n_out ]
                t_outputs  = saved[ -n_out: ]

                np_inputs      = [ t.cpu().detach().numpy() for t in t_inputs ]
                np_outputs     = [ t.cpu().detach().numpy() for t in t_outputs ]
                np_grads       = [ g.cpu().detach().numpy() for g in grad_outputs ]
                np_grad_inputs = [ np.zeros( t.shape, dtype = np_dtype ) for t in t_inputs ]

                backward_func( *np_grad_inputs, *np_outputs, *np_grads, *np_inputs )

                return tuple( torch.as_tensor( g, dtype = outer_self.dtype, device = outer_self.device ) for g in np_grad_inputs )

        return Func.apply( *input_tensors )


    def array_conversion( self, value ):
        """ Ensure that every array of value is of the right type """
        if isinstance( value, self.array_type ):
            return value
        if isinstance( value, np.ndarray ):
            return self.tn( value )
        if isinstance( value, list ):
            return [ self.array_conversion( v ) for v in value ]
        if isinstance( value, tuple ):
            return tuple( self.array_conversion( v ) for v in value )
        if hasattr( value, "array_conversion" ):
            return value.array_conversion()
        raise NotImplementedError


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

        input_tensors = flat_tensor_list( f ) + flat_tensor_list( g )
        outputs = SDOTFunction.apply( *input_tensors )
        assert isinstance( outputs, tuple )
        return BatchOfOtPlans( *outputs )

    def optimize_using_lbfgs( self, loss, params, max_iter=50, tol_grad=1e-7, on_iter=None ):
        """ small helper to optimize `loss` wrt `params` using L-BFGS.
            - `params`  : torch tensor or list of torch tensors
            - `on_iter` : optional callback( params, iter, grad_norm ) called each iteration
            Returns the optimized params (same type as input).
        """
        is_list = isinstance( params, ( list, tuple ) )
        p_list  = list( params ) if is_list else [ params ]

        for p in p_list:
            p.requires_grad_( True )

        lbfgs = torch.optim.LBFGS( p_list, history_size=15, max_iter=10 )

        def closure():
            lbfgs.zero_grad()
            objective = loss( p_list if is_list else p_list[ 0 ] )
            objective.backward()
            return objective

        for i in range( max_iter ):
            lbfgs.step( closure )

            with torch.no_grad():
                grad_norm = float( torch.norm( torch.stack( [ torch.norm( p.grad ) for p in p_list if p.grad is not None ] ) ) )

                if on_iter:
                    on_iter( p_list if is_list else p_list[ 0 ], i, grad_norm )

                if grad_norm < tol_grad:
                    break

        return p_list if is_list else p_list[ 0 ]

    def optimize_using_sgd( self, loss, params ):
        optimizer = torch.optim.SGD( [ params ] )
        for _ in range( 20 ):
            l = loss( params )
            print( l )
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
