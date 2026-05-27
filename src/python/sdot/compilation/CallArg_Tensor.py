from ..util import append_if_unique
from ..aggregate.AxisExpr import AxisExpr
from ..drivers.driver import driver
from ..drivers.Dtype import Dtype

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

    ct_variables              : list[ str ]

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

    orig_parent               : Optional[ ref ]

    def __init__(
        self,
        name_in_parent            : Optional[ str ],
        parent                    : Optional[ ref ],
        python_class              : any,
        python_value              : Optional[ any ],
        io_category               : IoCategory,
        ctor_args                 : Optional[ list ],
        ctor_kwargs               : Optional[ dict ],
        shape                     : list[ AxisExpr ],
        dtype                     : Dtype,
        ct_variables              : list[ str ],
        represents_a_dynamic_axis : str = "",
        comes_from_basic_array    : bool = False
    ):
        super().__init__( name_in_parent, parent, python_class, python_value, io_category, ctor_args, ctor_kwargs )

        self.represents_a_dynamic_axis = represents_a_dynamic_axis
        self.comes_from_basic_array    = comes_from_basic_array
        self.is_differentiable         = dtype.floating_point
        self.shape                     = shape
        self.dtype                     = dtype
        self.ct_variables              = ct_variables

        self.validity_output_index     = -1
        self.tensor_type_input_index   = -1

        self.num_in_input_sub_list     = -1
        self.num_in_outputs            = -1
        self.num_in_dynamic_axes       = -1

        self.orig_parent               = None

    @staticmethod
    def factory( call_args, parent, name_in_parent, python_class, python_value, io_category: IoCategory, ctor_args, ctor_kwargs, shape: Optional[ list[ AxisExpr ] ] = None, dtype = None, ct_variables = None, represents_a_dynamic_axis = "", comes_from_basic_array = False ):
        """  """
        if ct_variables is None:
            ct_variables = []

        if shape is None:
            if python_value is not None:
                orig = python_value.shape
            else:
                orig = python_class.shape
            shape = [ AxisExpr( s ) for s in list( orig ) ]

        if dtype is None and python_value is not None:
            dtype = Dtype.factory( python_value.dtype )
        assert isinstance( dtype, Dtype )

        res = CallArg_Tensor(
            name_in_parent            = name_in_parent,
            parent                    = parent,
            python_class              = python_class,
            python_value              = python_value,
            io_category               = io_category,
            ctor_args                 = ctor_args,
            ctor_kwargs               = ctor_kwargs,
            shape                     = shape,
            dtype                     = dtype,
            ct_variables              = ct_variables,
            represents_a_dynamic_axis = represents_a_dynamic_axis,
            comes_from_basic_array    = comes_from_basic_array
        )

        if call_args and represents_a_dynamic_axis:
            res.num_in_dynamic_axes = len( call_args.dynamix_axes )
            call_args.dynamix_axes.append( res )

        # input or mutable -> need an input tensor
        if io_category.has_input:
            res.tensor_type_input_index = call_args.get_u8_input( [ CallArg_Tensor.tensor_type_index( python_value ) ] )
            res.num_in_input_sub_list = call_args.add_tensor_input( res )

        # mutable, return or workspace -> need an output tensor
        if io_category.want_output:
            res.validity_output_index = call_args.get_u8_input( [ TENSOR_TYPE_STD ] )
            res.num_in_outputs = call_args.add_tensor_output( res )

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

        # if in Workspace, return None
        if not self.io_category.want_return:
            return None

        # Trim each axis to its resolved (possibly dynamic) size. When the size cannot
        # be resolved (e.g. a traced value inside a JAX JIT), keep the full buffer.
        slices = []
        for expr in self.shape:
            try:
                slices.append( slice( 0, int( expr.value( self.value_of_axis_variable, True ) ) ) )
            except Exception:
                slices.append( slice( None ) )

        return res[ tuple( slices ) ]



    def shape_values( self, use_dyn_size = False ):
        res = []
        for expr in self.shape:
            res.append( expr.value( self.value_of_axis_variable, use_dyn_size ) )
        return res

    @property
    def ndim( self ) -> int:
        res = 0
        for s in self.shape:
            res += s.ndim( self.value_of_axis_variable )
        return res

    def signature( self ) -> str:
        ct_values = []
        for ct_variable in self.ct_variables:
            ct_values.append( f"{ ct_variable }_{ self.value_of_axis_variable( ct_variable, False ) }" )
        return f"T{ self.ndim }{ Dtype.factory( self.dtype ).cpp_name }{ '-'.join( ct_values ) }"

    def get_template_args( self, template_args, names ):
        for name in self.ct_variables:
            lst = names[ : -1 ] + [ name ]
            template_args.add( f"ct_{ '_'.join( lst ) }", "TI", 4, str( self.value_of_axis_variable( name, False ) ) )

        if self.dtype.floating_point and self.dtype.size is None:
            template_args.add( "TF", "typename", 0, "TF" )

        template_args.add( "TI", "typename", 1, "TI" ) # always needed

        template_args.add( "MemorySpace", "typename", 2, "DECAYED_TYPE_OF( memory_space )" )


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
        """True if axis n is declared compile-time (all terms in ct_variables, no dynamic selection)."""
        expr = self.shape[ n ]
        for term in expr.terms:
            if term.variable.selection is not None or term.variable.arguments is not None:
                return False
            if term.variable.name not in self.ct_variables:
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
            return int( expr.value( self.value_of_axis_variable, False ) )
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
        shape_t = self.shape_type( False )
        if self.represents_a_dynamic_axis:
            return f"DynamicAxis<TI,MemorySpace,{ shape_t }>"
        return f"TensorView<{ self.dtype.cpp_name },MemorySpace,{ shape_t }>"

    def get_ct_axis_variable_names( self, ct_axis_variable_names: list[ str ], name_list_so_far: list[ str ] ):
        for name in self.ct_variables:
            complete_name = "_".join( name_list_so_far[ : -1 ] + [ name ] )
            append_if_unique( ct_axis_variable_names, complete_name )

    def get_axis_variable_names( self, axis_variable_names: list[ str ] ):
        for s in self.shape:
            s.get_axis_variable_names( axis_variable_names )

    def add_axis_tensor_sources( self, sources, attributes, use_attributes, recursive ):
        from ..aggregate.AxisVariableSystem import AxisTensorSource

        cpp_ref = ".".join( attributes ) if use_attributes else "t_" + self.ffi_name()
        sources.append( AxisTensorSource(
            shape        = self.shape,
            ct_variables = self.ct_variables,
            numpy_value  = self.python_value,
            cpp_ref      = cpp_ref
        ) )

    def ffi_output_name( self ):
        return f"out{ self.num_in_outputs }"

    def ffi_input_name( self ):
        bn = "din" if self.is_differentiable else "nin"
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
            # the capacity is the maximum size the consuming tensors give to this axis
            assert self.parent
            capa = self.parent().axis_system().cpp_capacity_expr( "max_of_" + self.represents_a_dynamic_axis )
            extr = f", { self.num_in_dynamic_axes }, { capa }"
            base = "dynamic_axis"

        # mutable
        if self.io_category.has_input and self.io_category.want_output:
            return f"{ base }_mut( { ct_shape }, memory_space, { self.ffi_input_name() }, u8_input[ { self.tensor_type_input_index } ], { self.ffi_output_name() }, u8_input[ { self.validity_output_index } ]{ extr } )"

        # pure output
        if not self.io_category.has_input and self.io_category.want_output:
            return f"{ base }_out( { ct_shape }, memory_space, { self.ffi_output_name() }, u8_input[ { self.validity_output_index } ]{ extr } )"

        # pure input
        if self.io_category.has_input and not self.io_category.want_output:
            return f"{ base }_inp( { ct_shape }, memory_space, { self.ffi_input_name() }, u8_input[ { self.tensor_type_input_index } ]{ extr } )"

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

    def backward_version( self, call_args, driver, outputs, grads_of_the_outputs, parent, differentiable_inputs = None ):
        # something that was an output -> make an input
        python_value = self.python_value
        if self.io_category.want_output and self.num_in_outputs < len( outputs ):
            python_value = outputs[ self.num_in_outputs ]
        elif self.is_differentiable and differentiable_inputs is not None and 0 <= self.num_in_input_sub_list < len( differentiable_inputs ):
            python_value = differentiable_inputs[ self.num_in_input_sub_list ]

        res = CallArg_Tensor(
            name_in_parent            = self.name_in_parent,
            parent                    = parent,
            python_class              = self.python_class,
            python_value              = python_value,
            io_category               = IoCategory.pure_input(),
            ctor_args                 = self.ctor_args,
            ctor_kwargs               = self.ctor_kwargs,
            shape                     = self.shape,
            dtype                     = self.dtype,
            ct_variables              = self.ct_variables,
            represents_a_dynamic_axis = self.represents_a_dynamic_axis,
            comes_from_basic_array    = self.comes_from_basic_array
        )

        res.orig_parent = self.parent

        if call_args and res.represents_a_dynamic_axis:
            res.num_in_dynamic_axes = len( call_args.dynamix_axes )
            call_args.dynamix_axes.append( res )

        res.tensor_type_input_index = call_args.get_u8_input( [ CallArg_Tensor.tensor_type_index( python_value ) ] )
        res.num_in_input_sub_list = call_args.add_tensor_input( res )

        return res
