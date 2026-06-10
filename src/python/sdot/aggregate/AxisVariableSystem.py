from __future__ import annotations

from ..util import append_if_unique, index
from .AxisExpr import AxisExpr

from typing import Optional, cast


class AxisVariableSystem:
    """
    The single place where axis variables are resolved, for one aggregate scope.

    """

    def __init__( self, aggregate, explicit_values: Optional[ dict ] = None ):
        self.explicit_values = dict( explicit_values or {} )
        self.aggregate = aggregate

    def value_of( self, name: str, forbidden_names = None ) -> int | list[ int ] | None:
        """Concrete value of `name` (int, or list for an expansion axis), else None."""

        # avoid infinite recursion
        if forbidden_names is None:
            forbidden_names = [ name ]
        elif name in forbidden_names:
            return None

        # in explicit_values ?
        if name in self.explicit_values:
            v = self.explicit_values[ name ]
            if isinstance( v, ( list, tuple ) ):
                return [ int( e ) for e in v ]
            return int( v )

        # using len( shape )
        for annotation, value in self.aggregate._aggregate_items().values():
            a_shape = getattr( annotation, "shape", None )
            v_shape = getattr( value, "shape", None )
            if a_shape is not None and v_shape is not None:
                axes_with_arguments = [] # list[ AxisExpr]
                for axis_expr in a_shape:
                    axis_expr = cast( AxisExpr, axis_expr )
                    arg = axis_expr.argument
                    if arg is not None:
                        axes_with_arguments.append( arg )
                if len( axes_with_arguments ) == 1:
                    expr = axes_with_arguments[ 0 ]
                    value, _ = expr.solve( name, [ len( v_shape ) - 1 ], 0, self, forbidden_names + [ name ] )
                    if value is not None:
                        return value

        # direct solve ?
        info( name )
        for annotation, value in self.aggregate._aggregate_items().values():
            a_shape = getattr( annotation, "shape", None )
            v_shape = getattr( value, "shape", None )
            if a_shape is not None and v_shape is not None:
                index_v_shape = 0
                for axis_expr in a_shape:
                    axis_expr = cast( AxisExpr, axis_expr )

                    res, off = axis_expr.solve( name, v_shape, index_v_shape, self, forbidden_names )
                    if res is not None:
                        return res

                    info( name, off )
                    if off is None:
                        return None
                    index_v_shape += off

        # TODO: system solve

        return None


    @classmethod
    def from_sources( cls, sources: list[ AxisTensorSource ], explicit_values: Optional[ dict ] = None ):
        """(legacy) Build a system from pushed `AxisTensorSource`s (the C++ codegen path)."""
        self = cls.__new__( cls )
        self.aggregate = None
        self.explicit_values = dict( explicit_values or {} )
        self._init_from_sources( list( sources ) )
        return self

    # -- lazy queries (aggregate path) ----------------------------------------

    def has( self, name: str ) -> bool:
        if self.aggregate is None:
            return name in self.names or name in self.explicit_values
        return ( name in self.explicit_values
                 or name in self.aggregate._axis_variable_names()
                 or name in self.aggregate._ct_axis_variable_names() )

    def check_consistency( self ):
        """Raise if two known tensor shapes imply different values for the same axis variable."""
        if self.aggregate is None:
            return self._sources_check_consistency()

        seen: dict = {}
        for name, equation in self.aggregate._axis_variable_equations():
            value = equation.direct_value()
            if value is None or isinstance( value, list ):
                continue
            if name in seen and seen[ name ] != value:
                raise ValueError( f"inconsistent value for axis variable '{ name }': { seen[ name ] } vs { value }" )
            seen[ name ] = value

    # -- legacy source machinery (C++ codegen path) ---------------------------

    def _init_from_sources( self, sources: list[ AxisTensorSource ] ):
        self.sources = sources

        # axis variable names (run-time + compile-time), in order of appearance
        self.ct_names = []
        self.names = []
        for src in self.sources:
            for expr in src.shape:
                expr.get_axis_variable_names( self.names )
            for name in src.ct_variables:
                append_if_unique( self.ct_names, name )
                append_if_unique( self.names, name )

        # equations: ( source, axis, row, offset )
        self.equations = []
        for src in self.sources:
            for axis, expr in enumerate( src.shape ):
                eq = expr.as_equation_row( self.names )
                if eq is None:
                    continue
                row, offset = eq
                self.equations.append( ( src, axis, row, offset ) )

    def _pins( self, num_axis: int, row: list[ int ] ) -> bool:
        """True if `row` determines axis `num_axis` alone (single non-zero coeff)."""
        return row[ num_axis ] != 0 and sum( 1 for c in row if c != 0 ) == 1

    def _sources_local_value_of( self, name: str ) -> Optional[ int ]:
        if name in self.explicit_values:
            return int( self.explicit_values[ name ] )

        num_axis = index( self.names, name )
        if num_axis < 0:
            return None

        for src, axis, row, offset in self.equations:
            if src.numpy_value is None or not self._pins( num_axis, row ):
                continue
            return ( int( src.numpy_value.shape[ axis ] ) - offset ) // row[ num_axis ]

        return None

    def _sources_check_consistency( self ):
        for num_axis, name in enumerate( self.names ):
            seen = None
            for src, axis, row, offset in self.equations:
                if src.numpy_value is None or not self._pins( num_axis, row ):
                    continue
                val = ( int( src.numpy_value.shape[ axis ] ) - offset ) // row[ num_axis ]
                if seen is None:
                    seen = val
                elif seen != val:
                    raise ValueError( f"inconsistent value for axis variable '{ name }': { seen } vs { val }" )

    # -- C++ codegen (legacy source path only) --------------------------------

    def cpp_runtime_expr( self, name: str ) -> str:
        """
        C++ expression evaluating to the run-time value of `name`, picking the first
        valid tensor that pins it. Compile-time tensors contribute `ct_<name>` directly.
        """
        cases = []

        for src in self.sources:
            if name in src.ct_variables:
                cases.append( f"ct_{ name }" )

        num_axis = index( self.names, name )
        if num_axis >= 0:
            for src, axis, row, offset in self.equations:
                if src.cpp_ref is None or not self._pins( num_axis, row ):
                    continue
                coeff = row[ num_axis ]
                op = f"{ src.cpp_ref }.shape( { axis } )"
                if offset:
                    op = f"( { op } - { offset } ) / { coeff }" if coeff != 1 else f"{ op } - { offset }"
                elif coeff != 1:
                    op = f"{ op } / { coeff }"
                # is_valid() is always compile-time in the FFI binding context (TensorView/ZeroTensor
                # from the binding helpers are always valid; NoneTensor sources are excluded upstream).
                # cases.append( op )
                return op

        # return f"first_positive( \"{ name }\", { ', '.join( cases ) } )"
        # if not cases:
        return "0"

    def cpp_capacity_expr( self, dynamic_axis_name: str ) -> str:
        """C++ expression for the capacity of a `DynamicAxis` (same resolution as a plain axis)."""
        return self.cpp_runtime_expr( dynamic_axis_name )
