from sdot.compilation.CallArg_Tensor import CallArg_Tensor
from sdot.aggregate.ShapeItem import ShapeItem
from sdot.util.find import find
from ..driver import driver

# from .UndefinedTensor import UndefinedTensor

# from typing import Self, overload, TYPE_CHECKING
import numpy

class GenericTensor( numpy.ndarray ):
    """ created only to please typing """
    requires_grad : bool


class Tensor:
    """
    Descriptor for a tensor in an aggrate, Return of Workspace (or BatchOfDistributions, ...).

    axis_names : names of each axis (e.g. "nb_diracs", "dim").

    The tensor rank equals len( axis_names ) excepted if there's a "*dim" in the axis names

    If the field value is None and the host class defines a method
    ``default_<name>(self)``, that method is called to supply the value. ``default_<name>`` can take an additionnal argument,
    ``default_<name>( self, batch_version )``

    In distribution, we store the tensor value in 'distribution._{ name }'.

    When we set a Tensor, we update 'distribution._{ name }' with a tensor compatible with the choices in sdot.driver
    """

    represents_a_dynamic_axis: bool
    comes_from_a_dim_list: bool
    removed_dim_axes: list[ int ]

    shape : list[ ShapeItem ]
    dtype: any
    name: str

    def __init__( self, *axes, dtype = None, represents_a_dynamic_axis = False ):
        self.represents_a_dynamic_axis = represents_a_dynamic_axis  # original args (with Dyn objects), used to reconstruct variants
        self.comes_from_a_dim_list = False
        self.removed_dim_axes = [] # used in variants like 1d version, ...

        self.shape = [ ShapeItem( axis ) for axis in axes ]
        self.dtype = dtype

    def __call__( self, array = None, dtype = None ):
        class TensorProxy:
            def __init__( self, array, dtype ):
                self.dtype = dtype
                self.__set__( array )
            def __get__( self, enclosing = None ):
                return self.array
            def __set__( self, array, enclosing = None ):
                self.array = driver.array( array, dtype = self.dtype )
        return TensorProxy( array, self.dtype )

    @property
    def ndim( self ):
        res = 0
        for s in self.shape:
            res += s.ndim
        return res

    def make_variant( self, batch_version: int, unidimensional_version: int ) -> 'Tensor':
        ct_axes = self.ct_axes.copy()
        ctor_args = []
        for name in self.axis_names:
            if ad := find( self.dynamic_axes, lambda x: x.name == name ):
                ctor_args.append( ad )
            else:
                ctor_args.append( name )

        if batch_version:
            new_ctor_args = [ "batch_size" ]
            for ctor_arg in ctor_args:
                if isinstance( ctor_arg, Dyn ):
                    if ctor_arg.capacity is not None:
                        raise NotImplementedError
                    new_ctor_args.append( Dyn(
                        one_value_for_each = [ "batch_size" ] + ctor_arg.one_value_for_each,
                        name = ctor_arg.name,
                    ) )
                else:
                    new_ctor_args.append( ctor_arg )
            ctor_args = new_ctor_args

        removed_dim_axes = self.removed_dim_axes.copy()
        if unidimensional_version:
            new_ctor_args = []
            for i, arg in enumerate( ctor_args ):
                name = arg.name if isinstance( arg, Dyn ) else arg
                if name == "dim":
                    removed_dim_axes.append( i )
                else:
                    new_ctor_args.append( arg )
            ctor_args = new_ctor_args

            if "dim" in ct_axes:
                del ct_axes[ "dim" ]

        new_field = Tensor( *ctor_args, dtype = self.dtype, represents_a_dynamic_axis = self.represents_a_dynamic_axis )
        new_field.comes_from_a_dim_list = self.comes_from_a_dim_list
        new_field.removed_dim_axes = removed_dim_axes
        new_field.ct_axes = ct_axes
        new_field.name = self.name
        return new_field


    def call_arg_factory( self, python_value, io_category: int ):
        return CallArg_Tensor.factory( self, python_value, io_category, self.shape, self.dtype )

    def get_axis_names( self, axis_names: set[ str ] ):
        for s in self.shape:
            s.get_axis_names( axis_names )
