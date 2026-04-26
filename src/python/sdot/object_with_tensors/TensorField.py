from ..drivers.compilation.CallArg import CallArg
from ..UndefinedTensor import UndefinedTensor
from ..driver import driver
from ..Dyn import Dyn

from typing import Self, overload
from inspect import signature
import numpy

class GenericTensor( numpy.ndarray ):
    """ created only to please typing """
    requires_grad : bool


class TensorField:
    """
    Descriptor for a tensor field in a Distribution (or BatchOfDistributions, ...).

    axis_names : names of each axis (e.g. "nb_diracs", "dim").

    The tensor rank equals len( axis_names ) excepted if there's a "*dim" in the axis names

    If the field value is None and the host class defines a method
    ``default_<name>(self)``, that method is called to supply the value. ``default_<name>`` can take an additionnal argument,
    ``default_<name>( self, batch_version )``

    In distribution, we store the tensor value in 'distribution._{ name }'.

    When we set a TensorField, we update 'distribution._{ name }' with a tensor compatible with the choices in sdot.driver
    """

    def __init__( self, *axis_names, dtype = None ):
        self._ctor_args = axis_names  # original args (with Dyn objects), used to reconstruct variants
        self.comes_from_a_dim_list = False
        self.removed_dim_axes = [] # used in variants like 1d version, ...
        self.dtype = dtype
        self.name = None

        self.dynamic_axis_names = []
        self.static_axis_names = []
        self.axis_names = []
        for axis_name in axis_names:
            if isinstance( axis_name, Dyn ):
                if not axis_name.name.isidentifier():
                    raise RuntimeError( f"dynamic axes do not support operations (for axis { axis_name.name })" )
                self.axis_names.append( axis_name.name + "_capacity" )
                self.dynamic_axis_names.append( axis_name.name )
            else:
                self.static_axis_names.append( axis_name )
                self.axis_names.append( axis_name )


    @property
    def ndim( self ):
        for axis_name in self.axis_names:
            if "*" in axis_name:
                raise NotImplementedError
        return len( self.axis_names )

    def __set_name__( self, enclosing, name ):
        self.name = name

    # overloads for typing
    @overload
    def __get__( self, enclosing: None, _type: type ) -> Self: ...
    @overload
    def __get__( self, enclosing: object, _type: type ) -> GenericTensor: ...

    def __get__( self, enclosing, _type = None ):
        # not in a Distribution ?
        if enclosing is None:
            return self

        # we have a value ?
        value = enclosing.__dict__.get( f'_{ self.name }' )
        if value is not None:
            return value

        # we have a default method ?
        default_method = getattr( type( enclosing ), f'default_{ self.name }', None )
        if default_method is not None:
            # make the new value
            sig = signature( default_method )
            if len( sig.parameters ) == 2:
                value = default_method( enclosing, isinstance( enclosing, BatchOfDistributions ) )
            else:
                value = default_method( enclosing )

            # register it
            self.__set__( enclosing, value )

            # return the normalized value
            return enclosing.__dict__[ f'_{ self.name }' ]

        # not found :(
        def get_value( attr ):
            return getattr( enclosing, attr, None )
        return UndefinedTensor( _shape( self, get_value ), self.dtype )

    def __set__( self, distribution, value ):
        if value is None:
            return

        # make the tensor
        tensor = driver.tn( value, _rank( distribution, self.axis_names ), self.name, self.dtype )

        # check the dimensions
        for axis_name in _axis_names( self.axis_names ):
            size = getattr( distribution, axis_name )
            if size is not None:
                currs = distribution._axis_count_for( self, tensor, axis_name, [] )
                for curr in currs:
                    if curr is not None and curr != size:
                        raise RuntimeError( f"tensor used to define the '{ self.name }' attribute is not of the correct size along the '{ axis_name }' axis (expecting { size }, provided tensor shape - minus potential offset - is { curr })" )

        # register the tensor
        distribution.__dict__[ f'_{ self.name }' ] = tensor

    def configure_call_ret_for( self, call_arg, fai, driver, *args, **kwargs ):
        shape_with_dyn = self.shape_for( mandatory = True, **kwargs )
        for dynamic_axis_name in self.dynamic_axis_names:
            index = self.axis_names.index( dynamic_axis_name + "_capacity" )
            assert( index >= 0 )
            shape_with_dyn[ index ] = Dyn( dynamic_axis_name, shape_with_dyn[ index ] )

        list_of_dynamic_axes = None
        if call_arg.parent is not None:
            list_of_dynamic_axes = call_arg.parent().dynamic_axes

        call_arg.configure_as_output_tensor( fai, driver, shape_with_dyn, self.dtype or driver.dtype, resize_dyn_axes = False, list_of_dynamic_axes = list_of_dynamic_axes )

    def analysis_of_python_arg( self, python_value, name, fai, mutable, driver, parent = None ):
        res = CallArg.analysis_of_python_arg( python_value, name, fai, mutable, driver, parent )

        # add DynamicAxis members for input objects that declare dynamic axes
        for axis_name in self.dynamic_axis_names:
            num_axis = self.axis_names.index( axis_name + "_capacity" )
            assert( num_axis >= 0 )

            initial_value = getattr( parent.python_value, axis_name )
            fda = fai.add_dynamic_axis( axis_name, initial_value, parent.dynamic_axes, driver )
            if res.ffi_input:
                fda.add_input_capacity_source( res.ffi_input, num_axis, res.ffi_input.differentiable )

        return res

    def shape_for( self, mandatory = True, **kwargs ):
        def get_value( attr ):
            if attr not in kwargs:
                if mandatory:
                    raise RuntimeError( f"To get the shape of { self.name } we need the value of '{ attr }'" )
                return 0
            return kwargs[ attr ]
        return _shape( self, get_value )

    def make_variant( self, batch_version: int, unidimensional_version: int ) -> 'TensorField':
        ctor_args = list( self._ctor_args )

        if batch_version:
            ctor_args = [ "batch_size" ] + ctor_args

        removed_dim_axes = []
        if unidimensional_version:
            new_ctor_args = []
            for i, arg in enumerate( ctor_args ):
                name = arg.name if isinstance( arg, Dyn ) else arg
                if name == "dim":
                    removed_dim_axes.append( i )
                else:
                    new_ctor_args.append( arg )
            ctor_args = new_ctor_args

        new_field = TensorField( *ctor_args, dtype = self.dtype )
        new_field.removed_dim_axes = removed_dim_axes
        new_field.comes_from_a_dim_list = self.comes_from_a_dim_list
        return new_field

    def _rank( self, distribution ):
        return _rank( distribution, self.axis_names )

# ------------------------------------------------------ helpers ------------------------------------------------------
def _rank( distribution, base_axis_names ):
    res = 0
    for axis_name in base_axis_names:
        if "[" in axis_name:
            # _, axis = axis_name.split( "*" )
            # axis_size = getattr( distribution, axis )
            # if axis_size is None:
            #     return None

            # assert isinstance( axis_size, int )
            # res += axis_size
            # continue
            raise NotImplementedError

        res += 1

    return res

def _axis_value( text: str, get_value: callable ):
    if "+" in text:
        lhs, rhs = text.split( "+" )
        return _axis_value( lhs, get_value ) + _axis_value( rhs, get_value )

    if "*" in text:
        lhs, rhs = text.split( "*" )
        return _axis_value( lhs, get_value ) * _axis_value( rhs, get_value )

    #
    text = text.strip()
    if text.isdigit():
        return int( text )

    return get_value( text )


def _shape( field, get_value: callable ):
    res = []
    for axis_name in field.axis_names:
        if "," in axis_name:
            raise NotImplementedError
        if "[" in axis_name:
            raise NotImplementedError
        res.append( _axis_value( axis_name, get_value ) )
    return res

def _axis_names( axis_names: tuple[ str, ... ] ) -> list[ str ]:
    res = []
    def concat( values ):
        for value in values:
            if value not in res:
                res.append( value )

    for axis_name in axis_names:
        axis_name = axis_name.strip()

        if "," in axis_name:
            for v in axis_name.split( "," ):
                concat( _axis_names( v ) )
            continue

        if "+" in axis_name:
            for v in axis_name.split( "+" ):
                concat( _axis_names( v ) )
            continue

        if "*" in axis_name:
            for v in axis_name.split( "*" ):
                concat( _axis_names( v ) )
            continue

        if "[" in axis_name:
            lhs, rhs = axis_name.split( "[" )
            assert rhs.endswith( ']' )
            rhs = rhs[ :-1 ]

            concat( _axis_names( lhs ) )
            concat( _axis_names( rhs ) )
            continue

        concat( [ axis_name ] )

    return res
