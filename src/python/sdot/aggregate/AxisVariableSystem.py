from __future__ import annotations

from .AxisExpr import AxisExpr

from typing import Optional, cast
import numpy


class AxisVariableSystem:
    """
    The single place where axis variables are resolved, for one aggregate scope.

    It is built from an *aggregate scope* exposing, by duck typing:

        _aggregate_items()  ->  dict { name: ( annotation, value ) }

    where `annotation` carries the declared `shape` (a list of `AxisExpr`) and `value`
    carries the concrete `shape` (a list of sizes). The size scalars are read and combined
    only through `-`, `*`, `//`, so the *same* resolution works whether they are plain ints
    (python-side resolution) or `CppScalar`s (C++ codegen), in which case `value_of` returns
    a `CppScalar` whose `str()` is the C++ expression. See `AxisExpr.solve`.

    `explicit_values` pins some variables up-front (e.g. compile-time axes, whose value is
    already known as an int even on the C++ codegen path).
    """

    def __init__( self, aggregate, explicit_values: Optional[ dict ] = None ):
        self.explicit_values = dict( explicit_values or {} )
        self.aggregate = aggregate

    def value_of( self, name: str, forbidden_names = None ):
        """Value of `name` (int / CppScalar, or a list for an expansion axis), else None."""

        # avoid infinite recursion
        if forbidden_names is None:
            forbidden_names = [ name ]
        elif name in forbidden_names:
            return None

        # in explicit_values ?
        if name in self.explicit_values:
            v = self.explicit_values[ name ]
            if isinstance( v, ( list, tuple ) ):
                return numpy.ndarray( [ int( e ) for e in v ] )
            return v if not isinstance( v, ( int, numpy.integer ) ) else int( v )

        # `name` is the argument of an expansion (e.g. `dim` in `shape( dim )`): its value is
        # the number of axes that expansion spans = total axes minus the plain (non-expansion) ones.
        for annotation, value in self.aggregate._aggregate_items().values():
            a_shape = getattr( annotation, "shape", None )
            v_shape = getattr( value, "shape", None )
            if a_shape is not None and v_shape is not None:
                axes_with_arguments = [ axis_expr.argument for axis_expr in a_shape if axis_expr.argument is not None ]
                if len( axes_with_arguments ) == 1:
                    expr = axes_with_arguments[ 0 ]
                    res, _ = expr.solve( name, [ len( v_shape ) - 1 ], 0, self, forbidden_names + [ name ] )
                    if res is not None:
                        return res

        # direct solve: walk each tensor's declared shape, matching it against the concrete one
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

                    if off is None:
                        break
                    index_v_shape += off

        # TODO: system solve (several coupled variables)

        return None

    def check_consistency( self ):
        """Raise if two tensor shapes of this scope imply different values for the same axis."""
        names: list[ str ] = []
        for annotation, _ in self.aggregate._aggregate_items().values():
            a_shape = getattr( annotation, "shape", None )
            if a_shape is not None:
                for axis_expr in a_shape:
                    axis_expr.get_axis_variable_names( names )

        for name in names:
            seen = None
            for annotation, value in self.aggregate._aggregate_items().values():
                a_shape = getattr( annotation, "shape", None )
                v_shape = getattr( value, "shape", None )
                if a_shape is None or v_shape is None:
                    continue
                index_v_shape = 0
                for axis_expr in a_shape:
                    res, off = axis_expr.solve( name, v_shape, index_v_shape, self, [ name ] )
                    if isinstance( res, int ):
                        if seen is not None and seen != res:
                            raise ValueError( f"inconsistent value for axis variable '{ name }': { seen } vs { res }" )
                        seen = res
                    if off is None:
                        break
                    index_v_shape += off
