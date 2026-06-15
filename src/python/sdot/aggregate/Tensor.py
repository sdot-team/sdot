from ..compilation.CallArg_Tensor import CallArg_Tensor
from ..compilation.IoCategory import IoCategory

from ..util.append_if_unique import append_if_unique

from ..drivers.driver import driver
from ..drivers.Dtype import Dtype

from .AxisExpr import AxisExpr

import numpy

class GenericTensor( numpy.ndarray ):
    """ created only to please typing """
    requires_grad : bool


class Tensor:
    """
    Descriptor for a tensor in an aggrate (e.g. SplineGrid, BatchOfDistributions, ...), Return of Workspace

    axes : names of each axis (e.g. "nb_diracs", "dim").

    ct_variables is a list of
    - str -> name of axis variable
    - ( str, int ) -> name of axis variable with a limit (variable becomes runtime after that limit)
    """

    represents_a_dynamic_axis : str
    comes_from_a_dim_list     : bool
    removed_dim_axes          : list[ int ]

    ct_variables              : list
    shape                     : list[ AxisExpr ]
    dtype                     : Dtype
    name                      : str

    def __init__( self, *axis_exprs, dtype = None, ct_variables = None, represents_a_dynamic_axis = "" ):
        assert ct_variables is None or isinstance( ct_variables, list )

        self.represents_a_dynamic_axis = represents_a_dynamic_axis
        self.comes_from_a_dim_list = False
        self.removed_dim_axes = []

        self.ct_variables = list( ct_variables ) if ct_variables is not None else []
        self.shape = [ AxisExpr( s ) for s in axis_exprs ]
        self.dtype = Dtype.factory( dtype )

        # named batch axes as ( name, position ), or None when this Tensor declares none. Set by
        # make_variant when batch axes are prepended (e.g. a vmapped output) so the resulting
        # CallArg_Tensor can carry an AxisNames tag and be wrapped in BatchOf. See CallArg_Tensor.
        self.batch_axes = None


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

    def make_variant( self, additional_batch_axes: list[ str ], unidimensional_version: int ) -> 'Tensor':
        res = Tensor()
        res.represents_a_dynamic_axis = self.represents_a_dynamic_axis
        res.comes_from_a_dim_list = False
        res.removed_dim_axes = []
        res.dtype = self.dtype

        # the prepended axes are this variant's named batch axes, at leading positions; any axes the
        # source already declared shift right by that many.
        res.batch_axes = [ ( ax, i ) for i, ax in enumerate( additional_batch_axes ) ]
        if self.batch_axes:
            k = len( additional_batch_axes )
            res.batch_axes += [ ( name, pos + k ) for name, pos in self.batch_axes ]
        if not res.batch_axes:
            res.batch_axes = None

        # ct_variables
        if unidimensional_version:
            res.ct_variables = [ ct_variable for ct_variable in self.ct_variables if ct_variable != "dim" ]
        else:
            res.ct_variables = list( self.ct_variables )

        # shape
        res.shape = []
        for additional_batch_axis in additional_batch_axes:
            res.shape.append( AxisExpr( additional_batch_axis ) )
        for n, expr in enumerate( self.shape ):
            nexpr = expr
            if unidimensional_version:
                nexpr = nexpr.unidimensional_version()
                if nexpr.always_one and not expr.always_one:
                    res.removed_dim_axes.append( n )
                    continue

            res.shape.append( nexpr )

        return res

    def type_call_arg_factory( self, call_args, parent, name_in_parent, python_value, io_category: IoCategory, ctor_args, ctor_kwargs ):
        return CallArg_Tensor( call_args, parent, name_in_parent, self, python_value, io_category, ctor_args, ctor_kwargs, self.shape, self.dtype, self.ct_variables, represents_a_dynamic_axis = self.represents_a_dynamic_axis, batch_axes = getattr( self, "batch_axes", None ) )

    @property
    def ct_axis_variable_names( self ):
        # user variables
        res = [ v for v in self.ct_variables ]

        # add argument variables
        for expr in self.shape:
            for term in expr.terms:
                if term.arguments:
                    for argument in term.arguments:
                        for aterm in argument.terms:
                            if aterm.name not in self.ct_variables:
                                append_if_unique( res, aterm.name )

        return res

    @property
    def axis_variable_names( self ):
        res = []
        for s in self.shape:
            s.get_axis_variable_names( res )
        return res

