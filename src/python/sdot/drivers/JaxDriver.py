from ..BatchOfOtPlans import BatchOfOtPlans
from ..bindings import sdot_bindings_cpu
import jax.numpy as jnp
import jax
import numpy as np
import sdot

class JaxDriver:
    """
    JAX implementation for sdot centralization.
    """
    def __init__( self, dtype = jnp.float32, device = None ):
        self.device = device
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
        res = jnp.asarray( tensor, dtype = self.dtype )
        if self.device is not None:
             if isinstance( self.device, str ):
                 device = jax.devices( self.device )[ 0 ]
             else:
                 device = self.device
             res = jax.device_put( res, device )
        assert res.ndim == ndim
        return res

    def ones( self, shape ):
        res = jnp.ones( shape, dtype = self.dtype )
        if self.device is not None:
             if isinstance( self.device, str ):
                 device = jax.devices( self.device )[ 0 ]
             else:
                 device = self.device
             res = jax.device_put( res, device )
        return res

    def linspace( self, a, b, n ):
        res = jnp.linspace( a, b, n, dtype = self.dtype )
        if self.device is not None:
             if isinstance( self.device, str ):
                 device = jax.devices( self.device )[ 0 ]
             else:
                 device = self.device
             res = jax.device_put( res, device )
        return res

    def empty( self, shape ):
        # Following PyTorchDriver's choice of using ones for safety
        res = jnp.ones( shape, dtype = self.dtype )
        if self.device is not None:
             if isinstance( self.device, str ):
                 device = jax.devices( self.device )[ 0 ]
             else:
                 device = self.device
             res = jax.device_put( res, device )
        return res

    def repeat( self, tensor, shape ):
        return jnp.tile( tensor, shape )

    def batch_of_ot_plan_for_Piecewise1dAffineFunctions( self, f: sdot.BatchOfSumOfWeightedDiracs, g: sdot.BatchOfPiecewiseAffineFunction1d ) -> BatchOfOtPlans:

        @jax.custom_vjp
        def sdot_op( dirac_xs, dirac_ws, point_xs, point_ys ):
            def forward_callback( d_xs, d_ws, p_xs, p_ys ):
                # We use numpy to allocate memory for the C++ call
                distances = np.empty( d_xs.shape[ 0 : 1 ], dtype = np.float32 )
                barycenters = np.empty( d_xs.shape, dtype = np.float32 )
                potentials = np.empty( d_ws.shape, dtype = np.float32 )

                sdot_bindings_cpu.ot_plan_to_piecewise_affine_1d( d_xs, d_ws, p_xs, p_ys, distances, barycenters, potentials )

                return distances, barycenters, potentials

            res_distances, res_barycenters, res_potentials = jax.pure_callback(
                forward_callback,
                ( jax.ShapeDtypeStruct( ( dirac_xs.shape[ 0 ], ), jnp.float32 ),
                  jax.ShapeDtypeStruct( dirac_xs.shape, jnp.float32 ),
                  jax.ShapeDtypeStruct( dirac_ws.shape, jnp.float32 ) ),
                dirac_xs, dirac_ws, point_xs, point_ys
            )
            return res_distances, res_barycenters, res_potentials

        def sdot_op_fwd( dirac_xs, dirac_ws, point_xs, point_ys ):
            distances, barycenters, potentials = sdot_op( dirac_xs, dirac_ws, point_xs, point_ys )
            # We save the inputs, the barycenters and potentials for the backward pass
            return ( distances, barycenters, potentials ), ( dirac_xs, dirac_ws, point_xs, point_ys, barycenters, potentials )

        def sdot_op_bwd( res, grads ):
            dirac_xs, dirac_ws, point_xs, point_ys, barycenters, potentials = res
            grad_distance, grad_barycenters, grad_potentials = grads

            def backward_callback( g_dist, g_bary, g_pot, d_xs, d_ws, p_xs, p_ys, bary, pot ):
                grad_dirac_xs = np.empty( d_xs.shape, dtype = np.float32 )
                grad_dirac_ws = np.empty( d_ws.shape, dtype = np.float32 )
                grad_point_xs = np.empty( p_xs.shape, dtype = np.float32 )
                grad_point_ys = np.empty( p_ys.shape, dtype = np.float32 )

                sdot_bindings_cpu.backward_ot_plan_to_piecewise_affine_1d(
                    g_dist, g_bary, pot, bary, d_xs, d_ws, p_xs, p_ys,
                    grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys
                )

                return grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys

            res_grads = jax.pure_callback(
                backward_callback,
                ( jax.ShapeDtypeStruct( dirac_xs.shape, jnp.float32 ),
                  jax.ShapeDtypeStruct( dirac_ws.shape, jnp.float32 ),
                  jax.ShapeDtypeStruct( point_xs.shape, jnp.float32 ),
                  jax.ShapeDtypeStruct( point_ys.shape, jnp.float32 ) ),
                grad_distance, grad_barycenters, grad_potentials, dirac_xs, dirac_ws, point_xs, point_ys, barycenters, potentials
            )
            return res_grads

        sdot_op.defvjp( sdot_op_fwd, sdot_op_bwd )

        distances, barycenters, potentials = sdot_op( f._nd_positions(), f.weights, g.xs, g.ys )
        return BatchOfOtPlans( distances, barycenters, potentials )
