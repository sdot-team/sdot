from ..aggregate.AxisExpr import AxisExpr
from ..driver import driver
from ..util import index

from .IoCategory import IoCategory
from .CallArg import CallArg

from typing import Optional
import weakref

class CallArg_Tensor( CallArg ):
    """ input or mutable
    """

    python_class              : any #
    python_value              : any # optional tensor
    io_category               : IoCategory
    parent                    : CallArg #

    ctor_kwargs               : dict
    ctor_args                 : list

    is_differentiable         : bool
    shape                     : list[ AxisExpr ]
    dtype                     : any

    represents_a_dynamic_axis : str

    validity_output_index     : int
    validity_input_index      : int

    num_in_input_sub_list     : int
    num_in_outputs            : int

    @staticmethod
    def factory( call_args, parent, name_in_parent, python_class, python_value, io_category: IoCategory, ctor_args, ctor_kwargs, shape: Optional[ list[ AxisExpr ] ] = None, dtype = None, ct_axes = [], represents_a_dynamic_axis = "" ):
        """  """
        if shape is None:
            orig = getattr( python_class, shape, None ) or python_value.shape
            shape = [ AxisExpr( s ) for s in orig ]

        if dtype is None:
            dtype = float

        res = CallArg_Tensor()

        res.python_class = python_class
        res.python_value = python_value
        res.io_category = io_category
        res.parent = weakref.ref( parent )

        res.ctor_kwargs = ctor_kwargs
        res.ctor_args = ctor_args

        res.is_differentiable = not driver.is_int_dtype( dtype )
        res.shape = shape
        res.dtype = dtype

        res.represents_a_dynamic_axis = represents_a_dynamic_axis

        res.validity_output_index = -1
        res.validity_input_index = -1

        res.num_in_input_sub_list = -1
        res.num_in_outputs = -1

        # input or mutable -> need an input tensor
        if io_category.has_input:
            res.validity_input_index = call_args.get_u8_input( [ CallArg_Tensor.is_valid( python_class ) ] )
            res.num_in_input_sub_list = call_args.add_tensor_input( res )

        # mutable, return or workspace -> need an output tensor
        if io_category.want_output:
            res.validity_output_index = call_args.get_u8_input( [ True ] )
            res.num_in_outputs = call_args.add_tensor_output( res )

        return res

    @staticmethod
    def is_valid( python_class ):
        return True

    @property
    def ffi_value( self ):
        if self.python_value is None:
            return driver.empty( [ 0 ] * self.ndim, dtype = self.dtype )
        return self.python_value

    @property
    def output_spec( self ):
        return driver.ffi_tensor_output_spec( self.shape_values(), self.dtype )

    def assemble_return( self, outputs ):
        if self.num_in_outputs > len( outputs ):
            return driver.empty( [ 0 ] * self.ndim, dtype = self.dtype )

        res = outputs[ self.num_in_outputs ]

        if

        return res

    def shape_values( self ):
        def get_axis_variable( name ):
            if name in self.ctor_kwargs:
                return int( self.ctor_kwargs[ name ] )
            info( self.ctor_kwargs )
            info( name )
            raise NotImplementedError

        res = []
        for expr in self.shape:
            res.append( expr.value( get_axis_variable ) )
        return res

    @property
    def ndim( self ) -> int:
        res = 0
        for s in self.shape:
            res += s.ndim()
        return res

    def signature( self ) -> str:
        return f"T{ self.ndim }{ driver.normalized_type_for( self.dtype ) }"

    def get_template_args( self, template_args ):
        if ( self.dtype is float ) or ( self.dtype is int ) or ( self.dtype is None ):
            template_args[ self.dtype_name() ] = "typename"
        template_args[ "Arch" ] = "typename"

    def cpp_type_name( self, main_list ):
        if self.represents_a_dynamic_axis:
            return f"DynamicAxis<{ self.dtype_name() },{ self.ndim },Arch>"
        return f"TensorView<{ self.dtype_name() },{ self.ndim },Arch>"

    def dtype_name( self ):
        if self.dtype is float or self.dtype is None:
            return "TF"
        if self.dtype is int:
            return "TI"
        return driver.normalized_type_for( self.dtype )

    def get_axes( self, axes: dict, ct_axes: dict[ int ] ):
        for s in self.shape:
            s.get_axes( axes, ct_axes )

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
        base = "tensor_view"
        extr = ""
        if self.represents_a_dynamic_axis:
            from .CallArg_Aggregate import CallArg_Aggregate
            p = self.parent()

            axes = {}
            ct_axes = {}
            for argument in p.sub_dict.values():
                argument.get_axes( axes, ct_axes )

            tensor_names, tensor_axes, matrix, vector = CallArg_Aggregate.axis_variable_equation( p.sub_dict, axes, False )
            axis_selection = axes[ self.represents_a_dynamic_axis ]

            capa = CallArg_Aggregate.get_axis_variable( axes, self.represents_a_dynamic_axis, axis_selection, tensor_names, tensor_axes, matrix, vector )
            numa = index( axes.keys(), self.represents_a_dynamic_axis )
            extr = f", { numa }, { capa }"
            base = "dynamic_axis"

        # mutable
        if self.io_category.has_input and self.io_category.want_output:
            return f"{ base }_mutable( CtInt<{ self.ndim }>(){ extr }, { self.ffi_input_name() }, u8_input[ { self.validity_input_index } ], { self.ffi_output_name() }, u8_input[ { self.validity_output_index } ] )"

        # pure output
        if not self.io_category.has_input and self.io_category.want_output:
            return f"{ base }_output( CtInt<{ self.ndim }>(){ extr }, { self.ffi_output_name() }, u8_input[ { self.validity_output_index } ] )"

        # pure input
        if self.io_category.has_input and not self.io_category.want_output:
            return f"{ base }_input( CtInt<{ self.ndim }>(){ extr }, { self.ffi_input_name() }, u8_input[ { self.validity_input_index } ] )"

        raise NotImplementedError


    def get_output_arg_decl( self, declarations: list ):
        declarations.append( f"{ driver.ffi_tensor_output_arg_code( self.ndim, self.dtype ) } { self.ffi_output_name() }" )

    def get_input_arg_decl( self, declarations: list ):
        declarations.append( f"{ driver.ffi_tensor_input_arg_code( self.ndim, self.dtype ) } { self.ffi_input_name() }" )

    def get_output_bind_chain( self, bind_chain: list ):
        bind_chain.append( driver.ffi_tensor_output_bind_code( self.ndim, self.dtype ) )

    def get_input_bind_chain( self, bind_chain: list ):
        bind_chain.append( driver.ffi_tensor_input_bind_code( self.ndim, self.dtype ) )

    def assembled_code( self, beg_line, parent ):
        return f"t_{ self.ffi_name() }"

        raise NotImplementedError
