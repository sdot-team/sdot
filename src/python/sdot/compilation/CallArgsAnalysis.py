from __future__ import annotations

from ..aggregate.Workspace import Workspace
from ..aggregate.Return import Return
from ..aggregate.Tensor import Tensor

from ..drivers.Dtype import Dtype

from .IoCategory import IoCategory

from .CallArg_Aggregate import CallArg_Aggregate
from .CallArg_Parameter import CallArg_Parameter
from .CallArg_Tensor import CallArg_Tensor


class CallArgsAnalysis:
    """
    """

    arguments : CallArg_Aggregate

    non_differentiable_tensor_inputs : list[ CallArg_Tensor ]
    differentiable_tensor_inputs : list[ CallArg_Tensor ]
    tensor_outputs : list[ CallArg_Tensor ]
    parameters : list[ CallArg_Parameter ]

    u8_input_values  : list[ int ]
    u64_output_size  : int

    index_dynamic_size_exception : int
    dynamix_axes : list

    def __init__( self, args : dict, parameter_class: str = "Parameters" ):
        # add tensors for output dynamic shapes
        nargs = args.copy()

        dynamic_shapes = {}
        source_kwargs  = {}
        for arg in nargs.values():
            if isinstance( arg, ( Return, Workspace ) ) and isinstance( arg.return_type, Tensor ):
                for expr in arg.return_type.shape:
                    for term in expr.terms:
                        if term.variable.selection is not None:
                            n = term.variable.name
                            dynamic_shapes[ n ] = term.variable.selection
                            source_kwargs.setdefault( n, {} ).update( getattr( arg, 'type_kwargs', {} ) )

        for name, selection in dynamic_shapes.items():
            if name not in nargs:
                nargs[ name ] = Workspace( Tensor( *selection, dtype = int, represents_a_dynamic_axis = name ), **source_kwargs.get( name, {} ) )

        # base parameters
        self.non_differentiable_tensor_inputs = []
        self.differentiable_tensor_inputs = []
        self.tensor_outputs = []
        self.parameters = []

        self.u8_input_values = []
        self.u64_output_size = 0

        self.index_dynamic_size_exception = self.get_u64_output( 2 )
        self.dynamix_axes = []

        self.parent = None

        # we make an object that behaves like an aggregate
        annotations = {}
        for name, argument in nargs.items():
            annotations[ name ] = type( argument )

        klass = type( parameter_class, (), { "__annotations__": annotations } )
        ainst = klass()
        for name, argument in nargs.items():
            setattr( ainst, name, argument )

        # make a CallArg_Aggregate
        self.arguments = CallArg_Aggregate( self, None, "p", klass, ainst, IoCategory( want_return = False, want_output = False, has_input = True, ), [], {} )

    def check_axis_consistency( self ):
        """Raise if tensors of a same aggregate disagree on a shared axis variable."""
        self.arguments.check_axis_consistency()

    @property
    def differentiable_ffi_inputs( self ):
        return [ fi.ffi_value for fi in self.differentiable_tensor_inputs ]

    @property
    def ffi_inputs( self ):
        from ..drivers.driver import driver

        import numpy
        res = [ fi.ffi_value for fi in self.differentiable_tensor_inputs + self.non_differentiable_tensor_inputs ]
        res.append( driver.array( self.u8_input_values, dtype = numpy.uint8 ) )
        return res

    @property
    def index_u64_output( self ):
        return len( self.tensor_outputs )

    @property
    def ffi_outputs( self ):
        from ..drivers.driver import driver

        res = [ fi.output_spec for fi in self.tensor_outputs ]
        res.append( driver.ffi_tensor_output_spec( [ self.u64_output_size ], Dtype.factory( "PI64" ) ) )
        return res

    @property
    def ffi_attributes( self ):
        res = {}
        for parameter in self.parameters:
            res[ parameter.name ] = parameter.python_value
        return res

    def update_differentiable_input_values_with( self, differentiable_input_values ):
        for n, value in enumerate( differentiable_input_values ):
            self.differentiable_tensor_inputs[ n ].python_value = value

    def update_objects( self, outputs ):
        for call_arg in self.tensor_outputs:
            if call_arg.io_category.want_output and call_arg.num_in_outputs < len( outputs ):
                call_arg.python_value = outputs[ call_arg.num_in_outputs ]

    def assemble_returns( self ):
        res = []
        for call_arg in self.arguments.sub_dict.values():
            if call_arg.io_category.want_return:
                res.append( call_arg.assemble_return() )
        return res

    def add_tensor_output( self, call_arg_tensor ):
        res = len( self.tensor_outputs )
        self.tensor_outputs.append( call_arg_tensor )
        return res

    def add_tensor_input( self, call_arg_tensor ):
        lst = self.non_differentiable_tensor_inputs
        if call_arg_tensor.is_differentiable:
            lst = self.differentiable_tensor_inputs
        res = len( lst )
        lst.append( call_arg_tensor )
        return res

    def add_parameter( self, call_arg_parameter ):
        res = len( self.parameters )
        self.parameters.append( call_arg_parameter )
        return res

    def get_u64_output( self, nb_u64 = 1 ):
        res = self.u64_output_size
        self.u64_output_size += nb_u64
        return res

    def get_u8_input( self, u8_values: list[ int ] ):
        assert isinstance( u8_values, list )
        res = len( self.u8_input_values )
        self.u8_input_values.extend( u8_values )
        return res

    def get_code_for_tensor_conversions( self, lines ):
        # regular tensors
        for tensor in self.non_differentiable_tensor_inputs + self.differentiable_tensor_inputs + self.tensor_outputs:
            if not tensor.represents_a_dynamic_axis:
                lines.append( f"    auto t_{ tensor.ffi_name() } = { tensor.ffi_conversion_code() };" )

        # dynamic sizes
        for tensor in self.non_differentiable_tensor_inputs + self.differentiable_tensor_inputs + self.tensor_outputs:
            if tensor.represents_a_dynamic_axis:
                lines.append( f"    auto t_{ tensor.ffi_name() } = { tensor.ffi_conversion_code() };" )

    def get_code_for_parameters_struct( self, includes: set, lines: list[ str ], struct_name: str ):
        self.arguments.struct_decl( struct_name, includes, lines )

    def arg_decl( self ) -> str:
        declarations = []

        # differentiable_tensor_inputs
        for inp in self.differentiable_tensor_inputs:
            inp.get_input_arg_decl( declarations )

        # non_differentiable_tensor_inputs
        for inp in self.non_differentiable_tensor_inputs:
            inp.get_input_arg_decl( declarations )

        # u8_input
        if len( self.u8_input_values ):
            from ..drivers.driver import driver
            declarations.append( driver.ffi_tensor_input_arg_code( 1, Dtype.factory( "PI8" ) ) + " u8_input_buffer" )

        # parameters
        for parameter in self.parameters:
            parameter.get_input_arg_decl( declarations )

        # outputs
        for out in self.tensor_outputs:
            out.get_output_arg_decl( declarations )

        # u64_output
        if self.u64_output_size:
            from ..drivers.driver import driver
            declarations.append( driver.ffi_tensor_output_arg_code( 1, Dtype.factory( "PI64" ) ) + " u64_output_buffer" )

        return ", ".join( declarations )

    def bind_chain( self ) -> list[ str ]:
        items = []

        for inp in self.differentiable_tensor_inputs:
            inp.get_input_bind_chain( items )

        for inp in self.non_differentiable_tensor_inputs:
            inp.get_input_bind_chain( items )

        if len( self.u8_input_values ):
            from ..drivers.driver import driver
            items.append( driver.ffi_tensor_input_bind_code( 1, Dtype.factory( "PI8" ) ) )

        for parameter in self.parameters:
            parameter.get_input_bind_chain( items )

        for out in self.tensor_outputs:
            out.get_output_bind_chain( items )

        if self.u64_output_size:
            from ..drivers.driver import driver
            items.append( driver.ffi_tensor_output_bind_code( 1, Dtype.factory( "PI64" ) ) )

        return items


    def backward_version( self, driver, outputs: list, grads_of_the_outputs, parameter_class: str, differentiable_inputs=None, perturbed_flags=None ):
        #
        res = CallArgsAnalysis( {}, parameter_class )
        res.arguments = self.arguments.backward_version( res, driver, outputs, grads_of_the_outputs, None, differentiable_inputs )

        # update python_class to get a correct base_cpp_name
        res.arguments.python_class = type( parameter_class, (), {} )

        # for each differentiable_tensor_inputs, add a grad output
        for i, ct in enumerate( self.differentiable_tensor_inputs ):
            name = "output_grad_for_" + ct.fully_qualified_name()
            needed = perturbed_flags is None or ( i < len( perturbed_flags ) and perturbed_flags[ i ] )
            grad_tensor = CallArg_Tensor(
                call_args = res,
                parent = res.arguments,
                name_in_parent = name,
                python_class = ct.python_class,
                python_value = None,
                io_category = IoCategory.for_return(),
                ctor_args = ct.ctor_args,
                ctor_kwargs = ct.ctor_kwargs,
                shape = ct.shape,
                dtype = ct.dtype,
                ct_variables = ct.ct_variables,
                represents_a_dynamic_axis = ct.represents_a_dynamic_axis,
                comes_from_basic_array = ct.comes_from_basic_array
            )
            if not needed:
                res.u8_input_values[ grad_tensor.validity_output_index ] = 0
            res.arguments.sub_dict[ name ] = grad_tensor

        # for each output, add a grad input
        for ct in self.tensor_outputs:
            name = "input_grad_for_" + ct.fully_qualified_name()
            python_value = None
            if ct.num_in_outputs < len( grads_of_the_outputs ):
                python_value = grads_of_the_outputs[ ct.num_in_outputs ]
            res.arguments.sub_dict[ name ] = CallArg_Tensor(
                call_args = res,
                parent = res.arguments,
                name_in_parent = name,
                python_class = ct.python_class,
                python_value = python_value,
                io_category = IoCategory.pure_input(),
                ctor_args = ct.ctor_args,
                ctor_kwargs = ct.ctor_kwargs,
                shape = ct.shape,
                dtype = ct.dtype,
                ct_variables = ct.ct_variables,
                represents_a_dynamic_axis = ct.represents_a_dynamic_axis,
                comes_from_basic_array = ct.comes_from_basic_array
            )

        return res
