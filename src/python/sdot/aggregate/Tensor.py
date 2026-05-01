from ..compilation.CallArg_Tensor import CallArg_Tensor
from ..compilation.IoCategory import IoCategory
from ..util.find import find
from .AxisExpr import AxisExpr
from ..driver import driver

import numpy

class GenericTensor( numpy.ndarray ):
    """ created only to please typing """
    requires_grad : bool


class Tensor:
    """
    Descriptor for a tensor in an aggrate (e.g. SplineGrid, BatchOfDistributions, ...), Return of Workspace

    axes : names of each axis (e.g. "nb_diracs", "dim").

    ct_axes is a list of
    - str -> name of ct_axis
    - ( str, int ) -> name of ct_axis with a limit (axis becomes runtime after that limit)
    """

    represents_a_dynamic_axis : str
    comes_from_a_dim_list : bool
    removed_dim_axes : list[ int ]

    ct_axes : list
    shape : list[ AxisExpr ]
    dtype : any
    name : str

    def __init__( self, *axis_exprs, dtype = None, ct_axes = [], represents_a_dynamic_axis = "" ):
        self.represents_a_dynamic_axis = represents_a_dynamic_axis
        self.comes_from_a_dim_list = False
        self.removed_dim_axes = []

        self.ct_axes = ct_axes
        self.shape = [ AxisExpr( s ) for s in axis_exprs ]
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

        new_field = Tensor( *ctor_args, dtype = self.dtype )
        new_field.comes_from_a_dim_list = self.comes_from_a_dim_list
        new_field.removed_dim_axes = removed_dim_axes
        new_field.ct_axes = ct_axes
        new_field.name = self.name
        return new_field


    def call_arg_factory( self, call_args, parent, python_value, io_category: IoCategory ):
        return CallArg_Tensor.factory( call_args, parent, self, python_value, io_category, self.shape, self.dtype, self.ct_axes, represents_a_dynamic_axis = self.represents_a_dynamic_axis )

    def get_axis_names( self, axis_names: set[ str ] ):
        for s in self.shape:
            s.get_axis_names( axis_names )
