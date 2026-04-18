import jax.numpy as jnp
import jax.core as jax_core
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

    @property
    def array_type( self ):
        return ( jax.Array, jax_core.Tracer )

    @property
    def int_type( self ):
        return jnp.int64

    def is_int_dtype( self, dtype ):
        return jnp.issubdtype( dtype, jnp.integer )

    def any_requires_grad( self, tensors ) -> bool:
        return True

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

    def tn( self, tensor, ndim, name = None, dtype = None ):
        """ make a rank ndim tensor """
        if tensor is None:
            return tensor

        if dtype is None or dtype is float:
            dtype = self.dtype
        elif dtype is int:
            dtype = self.int_type
        else:
            raise NotImplementedError( f"for { dtype }" )

        res = jnp.asarray( tensor, dtype = dtype, device = self.device )

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

    def empty( self, shape, dtype = None ):
        return jnp.zeros( shape, dtype = dtype or self.dtype, device = self.device )

    def expand_dims( self, tensor, index ):
        return jnp.expand_dims( tensor, index )

    def repeat( self, tensor, shape ):
        return jnp.tile( tensor, shape )

    def stack( self, tensors, axis ):
        return jnp.stack( tensors, axis=axis )

    def linalg_solve( self, A, b ):
        return jnp.linalg.solve( A, b )

    def moveaxis( self, tensor, source, destination ):
        return jnp.moveaxis( tensor, source, destination )

    def hstack( self, lst ):
        return jnp.hstack( lst )

    def to_numpy( self, t ):
        return np.array( t )
        # if isinstance( t, list ):
        # return t.to_numpy()

    def to_standard_objects( self, obj ):
        if isinstance( obj, jax.Array ):
            if self.is_int_dtype( obj.dtype ):
                return [ ( obj, "MI" ) ]
            return [ ( obj, "MF" ) ]
        # if isinstance( obj, jax_core.Tracer ):
        #     if self.is_int_dtype( obj.dtype ):
        #         return [ ( obj., "MI" ) ]
        #     return [ ( obj, "MF" ) ]
        return None

    def forward( self, forward_func, backward_func, fargs, input_tensors, output_args, output_tensors ):
        """Differentiable wrapper using pure_callback (safe for C++/nanobind).
            forward_func ( *fargs ) -> None  (fills output_tensors in-place via C++)
            backward_func( *fargs, *grad_inputs, *grad_outputs ) -> None
            New JAX arrays are written back into output_args via write_back_diffentiable_tensors.
        """
        np_dtype = np.dtype( self.dtype )
        n_out = len( output_tensors )

        # id-based mapping: which farg belongs to output_tensors / input_tensors / other JAX array
        out_id = { id( t ): i for i, t in enumerate( output_tensors ) }
        in_id = { id( t ): i for i, t in enumerate( input_tensors ) }

        def build_np_fargs( np_outputs, np_inputs ):
            result = []
            for a in fargs:
                if isinstance( a, ( jax.Array, jax_core.Tracer ) ) or id( a ) in out_id or id( a ) in in_id:
                    i_out = out_id.get( id( a ) )
                    i_in = in_id.get( id( a ) )
                    if i_out is not None:
                        result.append( np_outputs[ i_out ] )
                    elif i_in is not None:
                        result.append( np_inputs[ i_in ] )
                    else:
                        result.append( np.asarray( a ) )  # int / static tensors
                elif hasattr( a, "shape" ): # UndefinedTensor
                    shape = [ ( s if s is not None else 0 ) for s in a.shape ]
                    result.append( np.zeros( shape, dtype = a.dtype or np_dtype ) )
                else:
                    result.append( a )
            return result

        out_shapes = tuple( jax.ShapeDtypeStruct( t.shape, getattr( t, "dtype", self.dtype ) or self.dtype ) for t in output_tensors )
        inp_shapes = tuple( jax.ShapeDtypeStruct( t.shape, getattr( t, "dtype", self.dtype ) or self.dtype ) for t in input_tensors )

        @jax.custom_vjp
        def op( *jax_inputs ):
            def fwd_cb( *np_inputs ):
                from ..object_with_tensors.UndefinedTensor import UndefinedTensor
                np_outputs = [ np.empty( t.shape, dtype = t.dtype ) for t in out_shapes ]
                # We need to map UndefinedTensor to None for the nanobind call
                call_args = []
                for a in build_np_fargs( np_outputs, list( np_inputs ) ):
                    if isinstance( a, UndefinedTensor ):
                        call_args.append( None )
                    else:
                        call_args.append( np.asarray( a ) )
                forward_func( *call_args )
                return tuple( np_outputs )
            return jax.pure_callback( fwd_cb, out_shapes, *jax_inputs )

        def op_fwd( *jax_inputs ):
            outputs = op( *jax_inputs )
            return outputs, ( *outputs, *jax_inputs )

        def op_bwd( residuals, grad_outputs ):
            jax_outputs = residuals[ :n_out ]
            jax_inputs  = residuals[ n_out: ]

            def bwd_cb( *cb_args ):
                from ..object_with_tensors.UndefinedTensor import UndefinedTensor
                np_outputs      = list( cb_args[ :n_out ] )
                np_grad_outputs = list( cb_args[ n_out : 2 * n_out ] )
                np_inputs       = list( cb_args[ 2 * n_out: ] )
                np_grad_inputs  = [ np.zeros( t.shape, dtype = t.dtype ) for t in inp_shapes ]
                
                # We need to map UndefinedTensor to None for the nanobind call
                call_args = []
                for a in build_np_fargs( np_outputs, np_inputs ):
                    if isinstance( a, UndefinedTensor ):
                        call_args.append( None )
                    else:
                        call_args.append( np.asarray( a ) )
                        
                backward_func( *call_args, *np_grad_inputs, *np_grad_outputs )
                return tuple( np_grad_inputs )

            return jax.pure_callback( bwd_cb, inp_shapes,
                                      *jax_outputs, *grad_outputs, *jax_inputs )

        op.defvjp( op_fwd, op_bwd )

        jax_inputs  = [ jnp.asarray( t, dtype = getattr( t, "dtype", self.dtype ) or self.dtype, device = self.device ) for t in input_tensors ]
        new_outputs = op( *jax_inputs )

        # write new JAX arrays back into Output/Return objects (JAX arrays are immutable)
        new_tuple   = new_outputs if isinstance( new_outputs, tuple ) else ( new_outputs, )
        tracked_iter = iter( new_tuple )
        for arg in output_args:
            arg.write_back_diffentiable_tensors( tracked_iter )

        if n_out == 0:
            return None
        return new_tuple[ 0 ] if n_out == 1 else new_tuple

    def optimize_using_lbfgs( self, loss, params, max_iter=50, tol_grad=1e-7, on_iter=None ):
        """ small helper to optimize `loss` wrt `params` using L-BFGS (via scipy.optimize).
            - `params`  : JAX array or list of JAX arrays
            - `on_iter` : optional callback( params, iter, grad_norm ) called each iteration
            Returns the optimized params (same type as input).
        """
        import scipy.optimize

        # support single array or list of arrays
        is_list  = isinstance( params, ( list, tuple ) )
        p_list   = list( params ) if is_list else [ params ]
        shapes   = [ p.shape for p in p_list ]
        sizes    = [ int( np.prod( s ) ) for s in shapes ]

        def pack( arrays ):
            return np.concatenate( [ np.array( a ).flatten() for a in arrays ] ).astype( np.float64 )

        def unpack( x_flat ):
            parts, offset = [], 0
            for shape, size in zip( shapes, sizes ):
                parts.append( jnp.asarray( x_flat[ offset : offset + size ].reshape( shape ), dtype = self.dtype, device = self.device ) )
                offset += size
            return parts

        val_and_grad = jax.value_and_grad( lambda *ps: loss( list( ps ) if is_list else ps[ 0 ] ) )

        iter_ref = [ 0 ]

        def f_and_g( x_flat ):
            ps        = unpack( x_flat )
            val, grad = val_and_grad( *ps )
            g_list    = grad if isinstance( grad, ( list, tuple ) ) else [ grad ]
            return float( np.array( val ) ), pack( g_list )

        def callback( x_flat ):
            ps = unpack( x_flat )
            if on_iter:
                _, grad = val_and_grad( *ps )
                g_list   = grad if isinstance( grad, ( list, tuple ) ) else [ grad ]
                grad_norm = float( np.linalg.norm( pack( g_list ) ) )
                on_iter( ps if is_list else ps[ 0 ], iter_ref[ 0 ], grad_norm )
            iter_ref[ 0 ] += 1

        result = scipy.optimize.minimize(
            f_and_g, pack( p_list ),
            method  = 'L-BFGS-B',
            jac     = True,
            callback = callback,
            options  = { 'maxiter': max_iter, 'gtol': tol_grad },
        )

        final = unpack( result.x )
        return final if is_list else final[ 0 ]

    # def plan( self, bindings, f, g ):
    #     np_dtype = np.dtype( self.dtype )

    #     input_tensors = flat_tensor_list( f ) + flat_tensor_list( g )
    #     dirac_xs = input_tensors[ 0 ]

    #     batch_size = dirac_xs.shape[ 0 ]
    #     nb_diracs  = dirac_xs.shape[ 1 ]
    #     dim        = dirac_xs.shape[ 2 ]

    #     fwd_shapes = (
    #         jax.ShapeDtypeStruct( ( batch_size, ),                self.dtype ),
    #         jax.ShapeDtypeStruct( ( batch_size, nb_diracs, dim ), self.dtype ),
    #         jax.ShapeDtypeStruct( ( batch_size, nb_diracs ),      self.dtype ),
    #         jax.ShapeDtypeStruct( ( batch_size, nb_diracs, 2 ),   self.dtype ),
    #     )
    #     bwd_shapes = tuple(
    #         jax.ShapeDtypeStruct( t.shape, t.dtype ) for t in input_tensors
    #     )

    #     @jax.custom_vjp
    #     def sdot_op( *inputs ):
    #         def fwd_cb( *jax_inputs ):
    #             np_inputs = [ np.asarray( x ) for x in jax_inputs ]

    #             distances   = np.empty( ( batch_size, ),                dtype = np_dtype )
    #             barycenters = np.empty( ( batch_size, nb_diracs, dim ), dtype = np_dtype )
    #             potentials  = np.empty( ( batch_size, nb_diracs ),      dtype = np_dtype )
    #             cuts        = np.empty( ( batch_size, nb_diracs, 2 ),   dtype = np_dtype )

    #             binding_inputs = unflatten_args( f, g, np_inputs )
    #             bindings.forward( *binding_inputs, distances, barycenters, potentials, cuts )
    #             return distances, barycenters, potentials, cuts
    #         return jax.pure_callback( fwd_cb, fwd_shapes, *inputs )

    #     def sdot_op_fwd( *inputs ):
    #         outputs = sdot_op( *inputs )
    #         residuals = ( outputs[ 0 ], outputs[ 1 ], outputs[ 2 ], outputs[ 3 ], *inputs )
    #         return outputs, residuals

    #     def sdot_op_bwd( residuals, grads ):
    #         distances, barycenters, potentials, cuts = residuals[ 0 ], residuals[ 1 ], residuals[ 2 ], residuals[ 3 ]
    #         saved_inputs = residuals[ 4: ]
    #         grad_distances, grad_barycenters, grad_potentials, grad_cuts = grads

    #         def bwd_cb( *jax_args ):
    #             np_args = [ np.asarray( x ) for x in jax_args ]

    #             # Order in pure_callback call below: distances, barycenters, potentials, cuts, *saved_inputs, grad_distances, ...
    #             np_dist, np_bary, np_pot, np_cuts = np_args[ 0 ], np_args[ 1 ], np_args[ 2 ], np_args[ 3 ]
    #             n_in = len( input_tensors )
    #             np_inputs = np_args[ 4 : 4 + n_in ]
    #             np_grad_out = np_args[ 4 + n_in : ] # grad_distances, grad_barycenters, grad_potentials, grad_cuts

    #             flat_grad_inputs = [ np.zeros( t.shape, dtype = np_dtype ) for t in input_tensors ]

    #             binding_inputs = unflatten_args( f, g, np_inputs )
    #             binding_grad_inputs = unflatten_args( f, g, flat_grad_inputs )

    #             # bindings.backward expects: inputs..., distances, barycenters, potentials, cuts, grad_outputs..., grad_inputs...
    #             bindings.backward( *binding_inputs, np_dist, np_bary, np_pot, np_cuts, *np_grad_out, *binding_grad_inputs )

    #             return tuple( flat_grad_inputs )

    #         return jax.pure_callback(
    #             bwd_cb, bwd_shapes,
    #             distances, barycenters, potentials, cuts, *saved_inputs,
    #             grad_distances, grad_barycenters, grad_potentials, grad_cuts
    #         )

    #     sdot_op.defvjp( sdot_op_fwd, sdot_op_bwd )

    #     return BatchOfOtPlans( *sdot_op( *input_tensors ) )
