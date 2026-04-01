from ..BatchOfOtPlans import BatchOfOtPlans
from ..driver import driver
import jax.numpy as jnp
import numpy as np
import jax

map_of_plan_methods = {}

class JaxDriver:
    """
    JAX implementation for sdot centralization.
    """
    def __init__( self, normalized_dtype : str, normalized_device: str ):
        self.device = JaxDriver.find_device( normalized_device )
        self.dtype = JaxDriver.find_dtype( normalized_dtype )
        if normalized_dtype == "FP64":
            jax.config.update( "jax_enable_x64", True )

        self.map_of_plan_methods = map_of_plan_methods

    @property
    def name( self ) -> str:
        return "jax"

    @staticmethod
    def default_normalized_device_for( user_normalized_dtype ):
        platforms = { d.platform for d in jax.devices() }
        if "gpu" in platforms:
            return "cuda:0"
        # Metal (jax-metal) only supports FP32
        if "METAL" in platforms and user_normalized_dtype == "FP32":
            return "metal"
        return "cpu"

    @staticmethod
    def find_device( normalized_device: str ):
        """ find the jax device from a normalized name like cpu, cuda:1, metal """
        if normalized_device.startswith( "cpu" ):
            return jax.devices( "cpu" )[ 0 ]
        if normalized_device.startswith( "cuda" ):
            idx = int( normalized_device.split( ":" )[ 1 ] ) if ":" in normalized_device else 0
            return jax.devices( "gpu" )[ idx ]
        if normalized_device.startswith( "metal" ):
            return jax.devices( "METAL" )[ 0 ]
        raise RuntimeError( f"Unknown device { normalized_device }" )

    @staticmethod
    def find_dtype( normalized_dtype: str ):
        if normalized_dtype == "FP32":
            return jnp.float32
        if normalized_dtype == "FP64":
            return jnp.float64
        raise RuntimeError( f"Unknown dtype { normalized_dtype }" )

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
        res = jnp.asarray( tensor, dtype = self.dtype, device = self.device )
        if ndim is not None and res.ndim != ndim:
            if name is not None:
                raise IndexError( f"expecting for field '{ name }' a { ndim }d tensor, but { res.ndim }d was provided." )
            raise IndexError( f"expecting a { ndim }d tensor, but { res.ndim }d was provided." )
        return res

    def i1( self, tensor ):
        """ make a rank 1 int tensor """
        if tensor is None:
            return tensor
        res = jnp.asarray( tensor, dtype = jnp.int64, device = self.device )
        assert res.ndim == 1
        return res

    def ones( self, shape ):
        return jnp.ones( shape, dtype = self.dtype, device = self.device )

    def zeros( self, shape ):
        return jnp.zeros( shape, dtype = self.dtype, device = self.device )

    def linspace( self, a, b, n ):
        return jnp.linspace( a, b, n, dtype = self.dtype, device = self.device )

    def empty( self, shape ):
        return jnp.ones( shape, dtype = self.dtype, device = self.device )

    def expand_dims( self, tensor, index ):
        return jnp.expand_dims( tensor, index )

    def repeat( self, tensor, shape ):
        return jnp.tile( tensor, shape )

    def hstack( self, lst ):
        return jnp.hstack( lst )

    def to_numpy( self, t ):
        return np.array( t )
        # if isinstance( t, list ):
        # return t.to_numpy()

    def plan( self, bindings, f, g ):
        np_dtype = np.dtype( self.dtype )

        input_tensors = f.flat_tensor_list() + g.flat_tensor_list()
        dirac_xs = input_tensors[ 0 ]

        batch_size = dirac_xs.shape[ 0 ]
        nb_diracs  = dirac_xs.shape[ 1 ]
        dim        = dirac_xs.shape[ 2 ]

        fwd_shapes = (
            jax.ShapeDtypeStruct( ( batch_size, ),                self.dtype ),
            jax.ShapeDtypeStruct( ( batch_size, nb_diracs, dim ), self.dtype ),
            jax.ShapeDtypeStruct( ( batch_size, nb_diracs ),      self.dtype ),
            jax.ShapeDtypeStruct( ( batch_size, nb_diracs, 2 ),   self.dtype ),
        )
        bwd_shapes = tuple(
            jax.ShapeDtypeStruct( t.shape, t.dtype ) for t in input_tensors
        )

        @jax.custom_vjp
        def sdot_op( *inputs ):
            def fwd_cb( *np_inputs ):
                distances   = np.empty( ( batch_size, ),                dtype = np_dtype )
                barycenters = np.empty( ( batch_size, nb_diracs, dim ), dtype = np_dtype )
                potentials  = np.empty( ( batch_size, nb_diracs ),      dtype = np_dtype )
                cuts        = np.empty( ( batch_size, nb_diracs, 2 ),   dtype = np_dtype )
                bindings.forward( *np_inputs, distances, barycenters, potentials, cuts )
                return distances, barycenters, potentials, cuts
            return jax.pure_callback( fwd_cb, fwd_shapes, *inputs )

        def sdot_op_fwd( *inputs ):
            outputs = sdot_op( *inputs )
            residuals = ( outputs[ 1 ], outputs[ 2 ], outputs[ 3 ], *inputs )
            return outputs, residuals

        def sdot_op_bwd( residuals, grads ):
            n = len( input_tensors )
            barycenters, potentials, cuts = residuals[ 0 ], residuals[ 1 ], residuals[ 2 ]
            saved_inputs = residuals[ 3: ]
            grad_distances, grad_barycenters, grad_potentials, grad_cuts = grads

            def bwd_cb( *np_args ):
                np_bary, np_pot, np_cuts = np_args[ 0 ], np_args[ 1 ], np_args[ 2 ]
                np_inputs  = np_args[ 3 : 3 + n ]
                np_grad_in = np_args[ 3 + n : ]
                out_grads = tuple( np.empty( t.shape, dtype = np_dtype ) for t in input_tensors )
                bindings.backward( np_bary, np_pot, np_cuts, *np_inputs, *np_grad_in, *out_grads )
                return out_grads

            return jax.pure_callback(
                bwd_cb, bwd_shapes,
                barycenters, potentials, cuts, *saved_inputs,
                grad_distances, grad_barycenters, grad_potentials, grad_cuts
            )

        sdot_op.defvjp( sdot_op_fwd, sdot_op_bwd )

        return BatchOfOtPlans( *sdot_op( *input_tensors ) )

