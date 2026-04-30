from ..aggregate.ShapeItem import ShapeItem
from ..driver import driver
from ..util import index

from .CallArg import CallArg

from typing import Optional

class CallArg_Tensor( CallArg ):
    """ input or mutable
    """

    python_class: any #
    python_value: any # optional tensor
    io_category : int

    shape       : list[ ShapeItem ]
    dtype       : any

    # def configure_as_input_tensor( self, python_value: any, mutable: dict[ int ] | None, fai, driver, axis_names, ct_axes: dict[ int ], valid = True, represents_a_dynamic_axis = False ) -> Self:
    @staticmethod
    def factory( python_class, python_value, io_category: int, shape: Optional[ list[ ShapeItem ] ] = None, dtype = None ):
        """  """
        if shape is None:
            orig = getattr( python_class, shape, None ) or python_value.shape
            shape = [ ShapeItem( s ) for s in orig ]

        if dtype is None:
            dtype = float

        res = CallArg_Tensor()
        res.python_class = python_class
        res.python_value = python_value
        res.io_category = io_category
        res.shape = shape
        res.dtype = dtype
        return res

    def ndim( self ) -> int:
        res = 0
        for s in self.shape:
            res += s.ndim()
        return res

    def signature( self ) -> str:
        return f"T{ self.ndim() }{ driver.normalized_type_for( self.dtype ) }"

    def get_template_args( self, template_args ):
        if ( self.dtype is float ) or ( self.dtype is int ) or ( self.dtype is None ):
            template_args[ self.dtype_name() ] = "typename"
        template_args[ "Arch" ] = "typename"

    def cpp_type_name( self, main_list ):
        return f"TensorView<{ self.dtype_name() },{ self.ndim() },Arch>"

    def dtype_name( self ):
        if self.dtype is float or self.dtype is None:
            return "TF"
        if self.dtype is int:
            return "TI"
        return driver.normalized_type_for( self.dtype )

    def get_axes( self, axes: dict, ct_axes: dict[ int ] ):
        for s in self.shape:
            s.get_axes( axes, ct_axes )

    def get_all_the_ways_to_get( self, axis_names, attributes, tensor_names, tensor_axes, matrix, vector ):
        for n, s in enumerate( self.shape ):
            if len( s.terms ) == 0:
                continue

            tensor_names.append( ".".join( attributes ) )
            tensor_axes.append( n )

            row = [ 0 ] * len( axis_names )
            for term in s.terms:
                row[ index( axis_names, term.name ) ] = term.coeff

            vector.append( s.offset )
            matrix.append( row )

    def get_arg_decl( self, non_differentiable_inputs: list, differentiable_inputs: list, parameters: list, outputs: list ):
        if self.io_category <= 1: # input or mutable
            lst = differentiable_inputs
            bn = "di"
            if driver.is_int_dtype( self.dtype ):
                lst = non_differentiable_inputs
                bn = "ni"
            lst.append( f"{ driver.ffi_tensor_input_arg_code( self.ndim(), self.dtype ) } { bn }{ len( lst ) }" )

        if self.io_category >= 1: # mutable, return or workspace
            outputs.append( f"{ driver.ffi_tensor_output_arg_code( self.ndim(), self.dtype ) } o{ len( outputs ) }" )
