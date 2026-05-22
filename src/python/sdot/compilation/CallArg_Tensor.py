from ..aggregate.AxisExpr import AxisExpr
from ..drivers.driver import driver
from ..drivers.Dtype import Dtype
from ..util import index, find

from .IoCategory import IoCategory
from .CallArg import CallArg

from typing import Optional
from weakref import ref

TENSOR_TYPE_STD = 0,
TENSOR_TYPE_ZERO = 1,
TENSOR_TYPE_INVALID = 2,


class CallArg_Tensor( CallArg ):
    """ input or mutable
    """

    ct_axes                   : dict

    represents_a_dynamic_axis : str
    comes_from_basic_array    : bool
    is_differentiable         : bool
    shape                     : list[ AxisExpr ]
    dtype                     : Dtype

    tensor_type_input_index   : int
    validity_output_index     : int

    num_in_input_sub_list     : int
    num_in_dynamic_axes       : int
    num_in_outputs            : int

    @staticmethod
    def factory( call_args, parent, name_in_parent, python_class, python_value, io_category: IoCategory, ctor_args, ctor_kwargs, shape: Optional[ list[ AxisExpr ] ] = None, dtype = None, ct_axes = None, represents_a_dynamic_axis = "", comes_from_basic_array = False ):
        """  """
        if ct_axes is None:
            ct_axes = []

        if shape is None:
            if python_value is not None:
                orig = python_value.shape
            else:
                orig = python_class.shape
            shape = [ AxisExpr( s ) for s in list( orig ) ]

        if dtype is None and python_value is not None:
            dtype = Dtype.factory( python_value.dtype )
        assert isinstance( dtype, Dtype )

        res = CallArg_Tensor()

        # CallArg attributes
        res.name_in_parent = name_in_parent
        res.parent = ref( parent ) if parent is not None else None

        res.python_class = python_class
        res.python_value = python_value

        res.io_category = io_category

        res.ctor_kwargs = ctor_kwargs
        res.ctor_args = ctor_args

        # Tensor attributes
        res.represents_a_dynamic_axis = represents_a_dynamic_axis
        res.comes_from_basic_array = comes_from_basic_array
        res.is_differentiable = dtype.floating_point
        res.shape = shape
        res.dtype = dtype

        res.validity_output_index = -1
        res.tensor_type_input_index = -1

        res.num_in_input_sub_list = -1
        res.num_in_outputs = -1

        res.num_in_dynamic_axes = -1
        if call_args and represents_a_dynamic_axis:
            res.num_in_dynamic_axes = len( call_args.dynamix_axes )
            call_args.dynamix_axes.append( res )

        res.orig_parent = None

        # input or mutable -> need an input tensor
        if io_category.has_input:
            res.tensor_type_input_index = call_args.get_u8_input( [ CallArg_Tensor.tensor_type_index( python_value ) ] )
            res.num_in_input_sub_list = call_args.add_tensor_input( res )

        # mutable, return or workspace -> need an output tensor
        if io_category.want_output:
            res.validity_output_index = call_args.get_u8_input( [ TENSOR_TYPE_STD ] )
            res.num_in_outputs = call_args.add_tensor_output( res )

        #
        res.ct_axes = {}
        if ct_axes is not None:
            for name in ct_axes:
                assert isinstance( name, str )
                res.ct_axes[ name ] = None

        return res

    @staticmethod
    def tensor_type_index( python_value ):
        if driver.is_zero_tensor( python_value ):
            return TENSOR_TYPE_ZERO
        if python_value is None:
            return TENSOR_TYPE_INVALID
        return TENSOR_TYPE_STD

    @property
    def ffi_value( self ):
        if self.python_value is None or driver.is_zero_tensor( self.python_value ):
            return driver.empty( [ 0 ] * self.ndim, dtype = self.dtype )
        return self.python_value

    @property
    def output_spec( self ):
        return driver.ffi_tensor_output_spec( self.shape_values(), self.dtype )

    def assemble_return( self ):
        res = self.python_value
        if res is None:
            return res

        # if in Worspace, return None
        if not self.io_category.want_return:
            return None

        # if we have a dynamic size, make a slice
        # When inside a JAX JIT trace the size is a traced value (not concrete),
        # so we fall back to returning the full buffer without slicing.
        slices = []
        for expr in self.shape:
            axes = {}
            expr.get_axes( axes )
            da = find( axes.values(), lambda x: x is not None )
            if da is not None and len( da ) == 0:
                try:
                    slices.append( slice( 0, int( expr.value( self.get_axis_variable, True ) ) ) )
                except Exception:
                    slices.append( slice( None ) )
            else:
                slices.append( slice( None ) )

        res = res[ tuple( slices ) ]

        return res

    def get_axis_variable( self, name, is_a_dyn_size ):
        # Return( ..., dim = 2 )
        call_arg = self
        while True:
            if ck := getattr( call_arg, "ctor_kwargs", None ):
                if name in ck:
                    return int( ck[ name ] )
            if call_arg.parent is None:
                break
            call_arg = call_arg.parent()

        # orig_parent takes priority for ct_axes lookup (e.g. gradient tensors whose
        # parent is the backward aggregate, not the original one that holds the values)
        lookup_parent = self.orig_parent or self.parent

        # tensor
        if is_a_dyn_size and name in lookup_parent().sub_dict:
            return lookup_parent().sub_dict[ name ].python_value.item()

        #
        if enclosing := lookup_parent().python_value:
            return getattr( enclosing, name )

        raise RuntimeError( f"Unable to find '{ name }' in tensor '{ self.name_in_parent }', or in ctor args" )

    def shape_values( self, use_dyn_size = False ):
        res = []
        for expr in self.shape:
            res.append( expr.value( self.get_axis_variable, use_dyn_size ) )
        return res

    @property
    def ndim( self ) -> int:
        res = 0
        for s in self.shape:
            res += s.ndim( self.get_axis_variable )
        return res

    def signature( self ) -> str:
        ct_values = []
        for ct_variable in self.ct_axes.keys():
            ct_values.append( f"{ ct_variable }_{ self.get_axis_variable( ct_variable, False ) }" )
        return f"T{ self.ndim }{ Dtype.factory( self.dtype ).cpp_name }{ '-'.join( ct_values ) }"

    def get_template_args( self, template_args, names ):
        for name in self.ct_axes.keys():
            lst = names[ : -1 ] + [ name ]
            template_args.add( f"ct_{ '_'.join( lst ) }_value", "TI", 4 )

        if self.dtype.floating_point and self.dtype.size is None:
            template_args.add( "TF", "typename", 0 )

        template_args.add( "TI", "typename", 1 ) # always needed

        template_args.add( "MemorySpace", "typename", 2 )


    def _get_kwarg_only( self, name, is_dyn ):
        """Raise if `name` is not statically available from ctor_kwargs."""
        if is_dyn:
            raise KeyError( name )
        call_arg = self
        while True:
            if ck := getattr( call_arg, "ctor_kwargs", None ):
                if name in ck:
                    return int( ck[ name ] )
            if call_arg.parent is None:
                break
            call_arg = call_arg.parent()
        raise KeyError( name )

    def _is_ct_axis( self, n ) -> bool:
        """True if axis n is declared compile-time (all terms in ct_axes, no dynamic selection)."""
        expr = self.shape[ n ]
        for term in expr.terms:
            if term.variable.selection is not None or term.variable.arguments is not None:
                return False
            if term.variable.name not in self.ct_axes:
                return False
        return True

    def _ct_axis_value( self, n ):
        """Return the concrete compile-time value for axis n, or None if unavailable."""
        if not self._is_ct_axis( n ):
            return None
        expr = self.shape[ n ]
        if not expr.terms:
            return expr.offset
        try:
            return int( expr.value( self.get_axis_variable, False ) )
        except ( KeyError, RuntimeError, AttributeError, TypeError ):
            return None

    def shape_type( self, for_a_particular_binding ):
        """Build AxisTuple C++ type string with KnownAxisSize where axes are CT-known."""
        types = []
        for n in range( self.ndim ):
            if self._is_ct_axis( n ) and not self.comes_from_basic_array:
                types.append( f"Ct<TI,{ self._ct_axis_value( n ) }>" )
            else:
                types.append( "TI" )
                # if for_a_particular_binding:
                #     val = self._ct_axis_value( n )
                #     if val is not None:
                #         known.append( f"KnownAxisSize<TI,{ n },{ val }>" )
                # elif self._is_ct_axis( n ):
                #     expr = self.shape[ n ]
                #     ops = []
                #     if expr.offset:
                #        ops.append( str( expr.offset ) )
                #     for term in expr.terms:
                #         if term.coeff == 1:
                #             ops.append( f'ct_{ term.variable.name }_value' )
                #         else:
                #             ops.append( f'{ term.coeff } * ct_{ term.variable.name }_value' )
                #     val = ' + '.join( ops ) if ops else '0'
                #     known.append( f"KnownAxisSize<TI,{ n },{ val }>" )
        return f"Tuple<{ ','.join( types ) }>"

    def cpp_type_name( self, main_list ):
        shape_t   = self.shape_type( False )
        strides_t = f"DECAYED_TYPE_OF( contiguous_strides<{ self.dtype.cpp_name }>( { shape_t }() ) )"
        if self.represents_a_dynamic_axis:
            return f"DynamicAxis<TI,{ shape_t },{ strides_t },MemorySpace>"
        return f"TensorView<{ self.dtype.cpp_name },{ shape_t },{ strides_t },MemorySpace>"

    def get_axes( self, axes: dict, ct_axes: dict[ str, int ] ):
        for s in self.shape:
            s.get_axes( axes )

        for name, limit in self.ct_axes.items():
            ct_axes[ name ] = limit

    def get_all_the_ways_to_get( self, axis_names, attributes, use_attributes, tensor_names, tensor_axes, matrix, vector ):
        for n, s in enumerate( self.shape ):
            if len( s.terms ) == 0:
                continue

            if use_attributes:
                name = ".".join( attributes )
            else:
                name = "t_" + self.ffi_name()

            tensor_names.append( name )
            tensor_axes.append( n )

            row = [ 0 ] * len( axis_names )
            for term in s.terms:
                row[ index( axis_names, term.variable.name ) ] = term.coeff

            vector.append( s.offset )
            matrix.append( row )

    def ffi_output_name( self ):
        return f"o{ self.num_in_outputs }"

    def ffi_input_name( self ):
        bn = "di" if self.is_differentiable else "ni"
        return f"{ bn }{ self.num_in_input_sub_list }"

    def ffi_name( self ):
        if self.io_category.want_output:
            return self.ffi_output_name()
        return self.ffi_input_name()

    def ffi_conversion_code( self ):
        base      = "tensor_view"
        extr      = ""
        ct_shape  = f"CtType<{ self.shape_type( True ) }>()"

        if self.represents_a_dynamic_axis:
            p = self.parent()

            axes = {}
            ct_axes = {}
            for argument in p.sub_dict.values():
                argument.get_axes( axes, ct_axes )

            tensor_names, tensor_axes, matrix, vector = p.axis_variable_equation( axes, False )
            axis_selection = axes[ self.represents_a_dynamic_axis ]

            capa = p.get_axis_variable( axes, self.represents_a_dynamic_axis, axis_selection, tensor_names, tensor_axes, matrix, vector )
            extr = f", { self.num_in_dynamic_axes }, { capa }"
            base = "dynamic_axis"

        # mutable
        if self.io_category.has_input and self.io_category.want_output:
            return f"{ base }_mutable( { ct_shape }, memory_space, { self.ffi_input_name() }, u8_input[ { self.tensor_type_input_index } ], { self.ffi_output_name() }, u8_input[ { self.validity_output_index } ]{ extr } )"

        # pure output
        if not self.io_category.has_input and self.io_category.want_output:
            return f"{ base }_output( { ct_shape }, memory_space, { self.ffi_output_name() }, u8_input[ { self.validity_output_index } ]{ extr } )"

        # pure input
        if self.io_category.has_input and not self.io_category.want_output:
            return f"{ base }_input( { ct_shape }, memory_space, { self.ffi_input_name() }, u8_input[ { self.tensor_type_input_index } ]{ extr } )"

        raise NotImplementedError


    def get_output_arg_decl( self, declarations: list ):
        declarations.append( f"{ driver.ffi_tensor_output_arg_code( self.ndim, self.dtype ) } { self.ffi_output_name() }" )

    def get_input_arg_decl( self, declarations: list ):
        declarations.append( f"{ driver.ffi_tensor_input_arg_code( self.ndim, self.dtype ) } { self.ffi_input_name() }" )

    def get_output_bind_chain( self, bind_chain: list ):
        bind_chain.append( driver.ffi_tensor_output_bind_code( self.ndim, self.dtype ) )

    def get_input_bind_chain( self, bind_chain: list ):
        bind_chain.append( driver.ffi_tensor_input_bind_code( self.ndim, self.dtype ) )

    def assembled_code( self, beg_line ):
        return f"t_{ self.ffi_name() }"

    def backward_version( self, call_args, driver, outputs, grads_of_the_outputs, parent, differentiable_inputs=None ):
        res = CallArg_Tensor()
        self.init_CallArgs_backward_version( res, parent )

        res.orig_parent = self.parent
        res.ct_axes = self.ct_axes

        res.represents_a_dynamic_axis = self.represents_a_dynamic_axis
        res.comes_from_basic_array = self.comes_from_basic_array
        res.is_differentiable = self.is_differentiable
        res.shape = self.shape
        res.dtype = self.dtype

        res.validity_output_index = -1
        res.tensor_type_input_index = -1

        res.num_in_input_sub_list = -1
        res.num_in_outputs = -1

        res.num_in_dynamic_axes = -1
        if call_args and res.represents_a_dynamic_axis:
            res.num_in_dynamic_axes = len( call_args.dynamix_axes )
            call_args.dynamix_axes.append( res )

        # something that was an output -> make an input
        python_value = self.python_value
        if self.io_category.want_output and self.num_in_outputs < len( outputs ):
            python_value = outputs[ self.num_in_outputs ]
        elif self.is_differentiable and differentiable_inputs is not None and 0 <= self.num_in_input_sub_list < len( differentiable_inputs ):
            python_value = differentiable_inputs[ self.num_in_input_sub_list ]

        res.python_value = python_value
        res.tensor_type_input_index = call_args.get_u8_input( [ CallArg_Tensor.tensor_type_index( python_value ) ] )
        res.num_in_input_sub_list = call_args.add_tensor_input( res )

        return res
