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

    def tn( self, tensor, ndim ):
        """ make a rank ndim tensor """
        if tensor is None:
            return tensor
        res = jnp.asarray( tensor, dtype = self.dtype, device = self.device )
        assert res.ndim == ndim
        return res

    def ones( self, shape ):
        return jnp.ones( shape, dtype = self.dtype, device = self.device )

    def linspace( self, a, b, n ):
        return jnp.linspace( a, b, n, dtype = self.dtype, device = self.device )

    def empty( self, shape ):
        return jnp.ones( shape, dtype = self.dtype, device = self.device )

    def repeat( self, tensor, shape ):
        return jnp.tile( tensor, shape )

    def plan( self, f, g ):
        bindings = driver.bindings_for( f, g )
        np_dtype = np.dtype( self.dtype )

        input_tensors = f.tensor_list() + g.tensor_list()
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

    # def batch_of_ot_plan_for_Piecewise1dAffineFunctions( self, f: sdot.BatchOfSumOfWeightedDiracs, g: sdot.BatchOfPiecewiseAffineFunction1d ) -> BatchOfOtPlans:

    #     @jax.custom_vjp
    #     def sdot_op( dirac_xs, dirac_ws, point_xs, point_ys ):
    #         def forward_callback( d_xs, d_ws, p_xs, p_ys ):
    #             # We use numpy to allocate memory for the C++ call
    #             distances = np.empty( d_xs.shape[ 0 : 1 ], dtype = np.float32 )
    #             barycenters = np.empty( d_xs.shape, dtype = np.float32 )
    #             potentials = np.empty( d_ws.shape, dtype = np.float32 )

    #             sdot_bindings_cpu.ot_plan_to_piecewise_affine_1d( d_xs, d_ws, p_xs, p_ys, distances, barycenters, potentials )

    #             return distances, barycenters, potentials

    #         res_distances, res_barycenters, res_potentials = jax.pure_callback(
    #             forward_callback,
    #             ( jax.ShapeDtypeStruct( ( dirac_xs.shape[ 0 ], ), jnp.float32 ),
    #               jax.ShapeDtypeStruct( dirac_xs.shape, jnp.float32 ),
    #               jax.ShapeDtypeStruct( dirac_ws.shape, jnp.float32 ) ),
    #             dirac_xs, dirac_ws, point_xs, point_ys
    #         )
    #         return res_distances, res_barycenters, res_potentials

    #     def sdot_op_fwd( dirac_xs, dirac_ws, point_xs, point_ys ):
    #         distances, barycenters, potentials = sdot_op( dirac_xs, dirac_ws, point_xs, point_ys )
    #         # We save the inputs, the barycenters and potentials for the backward pass
    #         return ( distances, barycenters, potentials ), ( dirac_xs, dirac_ws, point_xs, point_ys, barycenters, potentials )

    #     def sdot_op_bwd( res, grads ):
    #         dirac_xs, dirac_ws, point_xs, point_ys, barycenters, potentials = res
    #         grad_distance, grad_barycenters, grad_potentials = grads

    #         def backward_callback( g_dist, g_bary, g_pot, d_xs, d_ws, p_xs, p_ys, bary, pot ):
    #             grad_dirac_xs = np.empty( d_xs.shape, dtype = np.float32 )
    #             grad_dirac_ws = np.empty( d_ws.shape, dtype = np.float32 )
    #             grad_point_xs = np.empty( p_xs.shape, dtype = np.float32 )
    #             grad_point_ys = np.empty( p_ys.shape, dtype = np.float32 )

    #             sdot_bindings_cpu.backward_ot_plan_to_piecewise_affine_1d(
    #                 g_dist, g_bary, pot, bary, d_xs, d_ws, p_xs, p_ys,
    #                 grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys
    #             )

    #             return grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys

    #         res_grads = jax.pure_callback(
    #             backward_callback,
    #             ( jax.ShapeDtypeStruct( dirac_xs.shape, jnp.float32 ),
    #               jax.ShapeDtypeStruct( dirac_ws.shape, jnp.float32 ),
    #               jax.ShapeDtypeStruct( point_xs.shape, jnp.float32 ),
    #               jax.ShapeDtypeStruct( point_ys.shape, jnp.float32 ) ),
    #             grad_distance, grad_barycenters, grad_potentials, dirac_xs, dirac_ws, point_xs, point_ys, barycenters, potentials
    #         )
    #         return res_grads

    #     sdot_op.defvjp( sdot_op_fwd, sdot_op_bwd )

    #     distances, barycenters, potentials = sdot_op( f._nd_positions(), f.weights, g.xs, g.ys )
    #     return BatchOfOtPlans( distances, barycenters, potentials )
