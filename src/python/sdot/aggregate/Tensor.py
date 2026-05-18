from ..compilation.CallArg_Tensor import CallArg_Tensor
from ..compilation.IoCategory import IoCategory
from ..drivers.driver import driver
from .AxisExpr import AxisExpr

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
    dtype : any # we do not use Dtype because the actual value will be specified by the driver
    name : str

    def __init__( self, *axis_exprs, dtype = None, ct_axes = None, represents_a_dynamic_axis = "" ):
        assert ct_axes is None or isinstance( ct_axes, list )

        self.represents_a_dynamic_axis = represents_a_dynamic_axis
        self.comes_from_a_dim_list = False
        self.removed_dim_axes = []

        self.ct_axes = list( ct_axes ) if ct_axes is not None else []
        self.shape = [ AxisExpr( s ) for s in axis_exprs ]
        self.dtype = dtype

        # add argument variables in ct_axes
        for expr in self.shape:
            for term in expr.terms:
                if term.variable.arguments:
                    for argument in term.variable.arguments:
                        for aterm in argument.terms:
                            if aterm.variable.name not in self.ct_axes:
                                self.ct_axes.append( aterm.variable.name )


    def __call__( self, array = None, dtype = None ):
        if array is None:
            return None
        return driver.array( array, dtype = dtype or self.dtype )

    def coerce( self, value ):
        if value is None:
            return None
        return driver.array( value, dtype = self.dtype )

    @property
    def ndim( self ):
        res = 0
        for s in self.shape:
            res += s.ndim
        return res

    def make_variant( self, batch_version: int, unidimensional_version: int ) -> 'Tensor':
        res = Tensor()
        res.represents_a_dynamic_axis = self.represents_a_dynamic_axis
        res.comes_from_a_dim_list = False
        res.removed_dim_axes = []
        res.dtype = self.dtype

        if unidimensional_version:
            res.ct_axes = [ ct_axis for ct_axis in self.ct_axes if ct_axis != "dim" ]
        else:
            res.ct_axes = list( self.ct_axes )

        res.shape = []
        if batch_version:
            res.shape.append( AxisExpr( "batch_size" ) )
        for n, expr in enumerate( self.shape ):
            nexpr = expr
            if unidimensional_version:
                nexpr = nexpr.unidimensional_version()
                if nexpr.always_one and not expr.always_one:
                    res.removed_dim_axes.append( n )
                    continue

            res.shape.append( nexpr )

        return res


    def call_arg_factory( self, call_args, parent, name_in_parent, python_value, io_category: IoCategory, ctor_args, ctor_kwargs ):
        return CallArg_Tensor.factory( call_args, parent, name_in_parent, self, python_value, io_category, ctor_args, ctor_kwargs, self.shape, self.dtype, self.ct_axes, represents_a_dynamic_axis = self.represents_a_dynamic_axis )

    def get_axis_names( self, axis_names: set[ str ] ):
        for s in self.shape:
            s.get_axis_names( axis_names )
