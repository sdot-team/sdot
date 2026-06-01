from ..util import append_if_unique
from ..aggregate.AxisExpr import AxisExpr
from ..drivers.driver import driver
from ..drivers.Dtype import Dtype

from .IoCategory import IoCategory
from .CallArg import CallArg

from typing import Optional
from weakref import ref


class CallArg_Tensor( CallArg ):
    """
    An array argument (input, output, or mutable). It owns the `shape` (a list of
    AxisExpr over axis variables) and the dtype, and registers itself in the flat FFI
    input/output lists. A tensor whose `represents_a_dynamic_axis` is set stands for a
    DynamicAxis rather than a plain TensorView.

    Input tensors map to one of three C++ types based on the python_value at call time:
      - TensorView  : normal array
      - ZeroTensor  : symbolic zero gradient (driver.is_zero_tensor is True)
      - NoneTensor  : absent / undefined gradient (python_value is None)
    The choice is baked into the generated C++ code (static dispatch), so each
    binding variant has a unique signature that includes the tensor kind prefix.
    """

    ct_variables              : list[ str ]

    represents_a_dynamic_axis : str
    comes_from_basic_array    : bool
    is_differentiable         : bool
    shape                     : list[ AxisExpr ]
    dtype                     : Dtype

    validity_output_index     : int  # index into u8_input_values for output-side validity

    num_in_input_sub_list     : int
    num_in_dynamic_axes       : int
    num_in_outputs            : int

    orig_parent               : Optional[ ref ]

    def __init__(
        self,
        call_args                 : any,
        parent                    : Optional[ ref ],
        name_in_parent            : Optional[ str ],
        python_class              : any,
        python_value              : Optional[ any ],
        io_category               : IoCategory,
        ctor_args                 : Optional[ list ],
        ctor_kwargs               : Optional[ dict ],
        shape                     : Optional[ list[ AxisExpr ] ] = None,
        dtype                     : Optional[ Dtype ] = None,
        ct_variables              : Optional[ list[ str ] ] = None,
        represents_a_dynamic_axis : str = "",
        comes_from_basic_array    : bool = False
    ):
        # value management: fill shape/dtype from the concrete array when not given
        if ct_variables is None:
            ct_variables = []
        if shape is None:
            orig = python_value.shape if python_value is not None else python_class.shape
            shape = [ AxisExpr( s ) for s in list( orig ) ]
        if dtype is None and python_value is not None:
            dtype = Dtype.factory( python_value.dtype )
        assert isinstance( dtype, Dtype )

        super().__init__( name_in_parent, parent, python_class, python_value, io_category, ctor_args, ctor_kwargs )

        self.represents_a_dynamic_axis = represents_a_dynamic_axis
        self.comes_from_basic_array    = comes_from_basic_array
        self.is_differentiable         = dtype.floating_point
        self.shape                     = shape
        self.dtype                     = dtype
        self.ct_variables              = ct_variables

        self.validity_output_index     = -1

        self.num_in_input_sub_list     = -1
        self.num_in_outputs            = -1
        self.num_in_dynamic_axes       = -1

        # analysis: register this tensor in the flat FFI input/output lists
        if call_args and represents_a_dynamic_axis:
            self.num_in_dynamic_axes = len( call_args.dynamix_axes )
            call_args.dynamix_axes.append( self )

        # NoneTensor and ZeroTensor pure inputs don't send an FFI buffer — no registration needed.
        # Mutable tensors (want_output=True) always register since the output buffer is real.
        if io_category.has_input and not self._is_none and not self._is_zero:
            self.num_in_input_sub_list = call_args.add_tensor_input( self )

        if io_category.want_output:  # mutable, return or workspace -> needs an output tensor
            self.validity_output_index = call_args.get_u8_input( [ 1 ] )
            self.num_in_outputs        = call_args.add_tensor_output( self )

    @property
    def _is_zero( self ) -> bool:
        """True when this is a pure input whose value is a symbolic zero gradient."""
        return not self.io_category.want_output and driver.is_zero_tensor( self.python_value )

    @property
    def _is_none( self ) -> bool:
        """True when this is a pure input whose value is absent (None)."""
        return not self.io_category.want_output and self.python_value is None

    @property
    def ffi_value( self ):
        """(value management) The array to send to the FFI call.

        Convention for special tensors:
          ZeroTensor  : prepend a marker dimension of size 0 → shape [0, s0, s1, ...]
          NoneTensor  : all-zero shape of the same rank → shape [0, 0, ..., 0]
        The C++ helpers zero_tensor_inp / none_tensor_inp match these conventions.
        """
        # _is_zero and _is_none (pure inputs) are never in the input list → ffi_value not called.
        # Only remaining case needing a dummy buffer: mutable tensor with python_value = None.
        if self.python_value is None:
            return driver.empty( [ 0 ] * self.ndim, dtype = self.dtype )

        return self.python_value

    @property
    def output_spec( self ):
        """(analysis) Shape+dtype spec of this tensor as an FFI output."""
        return driver.ffi_tensor_output_spec( self.shape_values(), self.dtype )

    def assemble_return( self ):
        """(value management, override) Return the output array, trimmed to its resolved sizes."""
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
        """(analysis) Concrete size of each axis, resolved through the axis variables."""
        res = []
        for expr in self.shape:
            res.append( expr.value( self.value_of_axis_variable, use_dyn_size ) )
        return res

    @property
    def ndim( self ) -> int:
        """(analysis) Number of numpy axes (an expansion axis counts for several)."""
        res = 0
        for s in self.shape:
            res += s.ndim( self.value_of_axis_variable )
        return res

    def signature( self ) -> str:
        """(analysis, override) Per-binding signature: kind prefix, rank, dtype, compile-time axes.

        Kind prefix: T=TensorView, Z=ZeroTensor, N=NoneTensor.
        """
        ct_values = []
        for ct_variable in self.ct_variables:
            ct_values.append( f"{ ct_variable }_{ self.value_of_axis_variable( ct_variable, False ) }" )
        if self._is_zero:
            prefix = "Z"
        elif self._is_none:
            prefix = "N"
        else:
            prefix = "T"
        return f"{ prefix }{ self.ndim }{ Dtype.factory( self.dtype ).cpp_name }{ '-'.join( ct_values ) }"

    def _concrete_cpp_type( self ) -> str:
        """Concrete C++ type for this tensor at the current call site (used as template arg value)."""
        shape_t = self.shape_type( False )
        if self._is_zero:
            return f"ZeroTensor<{ self.dtype.cpp_name },{ shape_t }>"
        if self._is_none:
            return f"NoneTensor<{ self.dtype.cpp_name }>"
        if self.represents_a_dynamic_axis:
            return f"DynamicAxis<TI,DECAYED_TYPE_OF( memory_space ),{ shape_t }>"
        return f"TensorView<{ self.dtype.cpp_name },DECAYED_TYPE_OF( memory_space ),{ shape_t }>"

    def get_template_args( self, template_args, names ):
        """(code generation, override) Emit the compile-time axis values, dtype, and tensor type parameter.

        Each tensor or DynamicAxis attribute contributes a `T_<path>` typename template parameter
        whose value at instantiation is the concrete C++ type (TensorView / ZeroTensor / NoneTensor / DynamicAxis).
        This keeps the generated struct generic across all tensor-kind and forward/backward combinations.
        """
        for name in self.ct_variables:
            lst = names[ : -1 ] + [ name ]
            template_args.add( f"ct_{ '_'.join( lst ) }", "TI", 4, str( self.value_of_axis_variable( name, False ) ) )

        # TI is needed for ct_<name> non-type template parameters (Ct<TI,V> in struct body)
        template_args.add( "TI", "typename", 1, "TI" )

        # Each attribute (tensor or DynamicAxis) becomes a generic type parameter T_<path>.
        # TF and MemorySpace are NOT added as standalone params: they are not referenced in the
        # generated struct body. User code can recover them via typename T_<name>::TF etc.
        param_name = "T_" + "_".join( names )
        template_args.add( param_name, "typename", 3, self._concrete_cpp_type() )

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
        """Build AxisTuple C++ type string with Ct<TI,V> where axes are CT-known."""
        types = []
        for n in range( self.ndim ):
            if self._is_ct_axis( n ) and not self.comes_from_basic_array:
                types.append( f"Ct<TI,{ self._ct_axis_value( n ) }>" )
            else:
                types.append( "TI" )
        return f"Tuple<{ ','.join( types ) }>"

    def cpp_type_name( self, names ):
        """(code generation, override) C++ member type for use inside a struct body.

        Returns the generic template parameter name T_<path> for all tensor and DynamicAxis
        attributes. The concrete type (TensorView / ZeroTensor / NoneTensor / DynamicAxis) is
        stored as the template arg value and resolved at instantiation time. This lets the same
        struct definition work for both forward (DynamicAxis) and backward (NoneTensor) passes.
        """
        return "T_" + "_".join( names )

    def get_ct_axis_variable_names( self, ct_axis_variable_names: list[ str ], name_list_so_far: list[ str ] ):
        """(analysis, override) Expose this tensor's compile-time axes, prefixed by the path."""
        for name in self.ct_variables:
            complete_name = "_".join( name_list_so_far[ : -1 ] + [ name ] )
            append_if_unique( ct_axis_variable_names, complete_name )

    def get_axis_variable_names( self, axis_variable_names: list[ str ] ):
        """(analysis, override) Expose the run-time axis variables appearing in the shape."""
        for s in self.shape:
            s.get_axis_variable_names( axis_variable_names )

    def add_axis_tensor_sources( self, sources, attributes, use_attributes, recursive ):
        """(analysis, override) Contribute this tensor (shape + value + C++ ref) to an AxisVariableSystem."""
        from ..aggregate.AxisVariableSystem import AxisTensorSource

        # NoneTensor has no shape to contribute
        if self._is_none:
            return

        if self._is_zero:
            # ZeroTensor has a shape but no FFI variable — use the inline expression as cpp_ref
            shape_t    = self.shape_type( True )
            shape_vals = self.shape_values()
            if not shape_vals:
                cpp_ref = f"ZeroTensor<{ self.dtype.cpp_name },{ shape_t }>()"
            else:
                shape_args = ", ".join( str( int( v ) ) for v in shape_vals )
                cpp_ref    = f"ZeroTensor<{ self.dtype.cpp_name },{ shape_t }>( { shape_t }( Values(), { shape_args } ) )"
        elif use_attributes:
            cpp_ref = ".".join( attributes )
        else:
            cpp_ref = "t_" + self.ffi_name()

        sources.append( AxisTensorSource(
            shape        = self.shape,
            ct_variables = self.ct_variables,
            numpy_value  = self.python_value,
            cpp_ref      = cpp_ref
        ) )

    def ffi_output_name( self ):
        """(code generation) Name of this tensor in the flat FFI output list."""
        return f"out{ self.num_in_outputs }"

    def ffi_input_name( self ):
        """(code generation) Name of this tensor in the flat FFI input list (differentiable or not)."""
        bn = "din" if self.is_differentiable else "nin"
        return f"{ bn }{ self.num_in_input_sub_list }"

    def ffi_name( self ):
        """(code generation) FFI name, taken from the output list for outputs else the input list."""
        if self.io_category.want_output:
            return self.ffi_output_name()
        return self.ffi_input_name()

    def ffi_conversion_code( self ):
        """(code generation) Build the C++ that wraps the raw FFI buffer(s) into a typed view.

        For inputs, the C++ type (TensorView / ZeroTensor / NoneTensor) is selected
        statically based on the python_value at binding-generation time.
        """
        ct_shape = f"CtType<{ self.shape_type( True ) }>()"
        base     = "dynamic_axis" if self.represents_a_dynamic_axis else "tensor_view"
        extr     = ""

        if self.represents_a_dynamic_axis:
            assert self.parent
            capa = self.parent().axis_system().cpp_capacity_expr( "max_of_" + self.represents_a_dynamic_axis )
            extr = f", { self.num_in_dynamic_axes }, { capa }"

        # mutable (has both input and output buffers)
        if self.io_category.has_input and self.io_category.want_output:
            if self._is_none:
                return f"{ base }_out( { ct_shape }, memory_space, { self.ffi_output_name() }{ extr } )"
            return f"{ base }_mut( { ct_shape }, memory_space, { self.ffi_output_name() }, { self.ffi_input_name() }{ extr } )"

        # pure output
        if not self.io_category.has_input and self.io_category.want_output:
            return f"{ base }_out( { ct_shape }, memory_space, { self.ffi_output_name() }{ extr } )"

        # pure input — static dispatch on tensor kind
        if self.io_category.has_input and not self.io_category.want_output:
            if self._is_zero:
                return f"zero_tensor_inp( { ct_shape }, { self.ffi_input_name() } )"
            if self._is_none:
                return f"none_tensor_inp<{ self.dtype.cpp_name }>()"
            return f"{ base }_inp( { ct_shape }, memory_space, { self.ffi_input_name() }{ extr } )"

        raise NotImplementedError

    def get_output_arg_decl( self, declarations: list ):
        """(code generation) Append the handler argument declaration for this output buffer."""
        declarations.append( f"{ driver.ffi_tensor_output_arg_code( self.ndim, self.dtype ) } { self.ffi_output_name() }" )

    def get_input_arg_decl( self, declarations: list ):
        """(code generation) Append the handler argument declaration for this input buffer."""
        declarations.append( f"{ driver.ffi_tensor_input_arg_code( self.ndim, self.dtype ) } { self.ffi_input_name() }" )

    def get_output_bind_chain( self, bind_chain: list ):
        """(code generation) Append the FFI binding for this output buffer."""
        bind_chain.append( driver.ffi_tensor_output_bind_code( self.ndim, self.dtype ) )

    def get_input_bind_chain( self, bind_chain: list ):
        """(code generation) Append the FFI binding for this input buffer."""
        bind_chain.append( driver.ffi_tensor_input_bind_code( self.ndim, self.dtype ) )

    def assembled_code( self, beg_line ):
        """(code generation, override) C++ expression for this tensor in a struct initializer.

        NoneTensor and ZeroTensor have no FFI buffer, so the expression is emitted inline.
        """
        if self._is_none:
            return f"none_tensor_inp<{ self.dtype.cpp_name }>()"
        if self._is_zero:
            shape_t    = self.shape_type( True )
            shape_vals = self.shape_values()
            if not shape_vals:
                return f"ZeroTensor<{ self.dtype.cpp_name },{ shape_t }>()"
            shape_args = ", ".join( str( int( v ) ) for v in shape_vals )
            return f"ZeroTensor<{ self.dtype.cpp_name },{ shape_t }>( { shape_t }( Values(), { shape_args } ) )"
        return f"t_{ self.ffi_name() }"

    def backward_version( self, call_args, driver, outputs, grads_of_the_outputs, parent, differentiable_inputs = None ):
        """(analysis, override) Mirror this tensor for the backward call: a former output becomes an input."""
        python_value = self.python_value
        if self.io_category.want_output and self.num_in_outputs < len( outputs ):
            python_value = outputs[ self.num_in_outputs ]
        elif self.is_differentiable and differentiable_inputs is not None and 0 <= self.num_in_input_sub_list < len( differentiable_inputs ):
            python_value = differentiable_inputs[ self.num_in_input_sub_list ]

        # __init__ registers the new (input) tensor in call_args' flat lists
        res = CallArg_Tensor(
            call_args                 = call_args,
            parent                    = parent,
            name_in_parent            = self.name_in_parent,
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

        return res
