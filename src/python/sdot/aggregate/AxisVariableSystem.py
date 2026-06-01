from __future__ import annotations

from ..util import append_if_unique, index
from .AxisExpr import AxisExpr
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AxisTensorSource:
    """
    One tensor contributing equations to an `AxisVariableSystem`.

    A source is produced the same way by an `@aggregate` instance (from its `Tensor`
    fields + attribute values) and by a `CallArg_Aggregate` (from its `sub_dict`), so
    the system itself never depends on `CallArg`.

    `shape`        declared axes, as a list of `AxisExpr`.
    `ct_variables` axis variable names known at compile time for this tensor.
    `numpy_value`  concrete array when available (python-side resolution), else None.
    `cpp_ref`      C++ expression to reach the tensor for run-time resolution
                   (e.g. "t_di0" in an FFI handler, or "positions" inside a struct).
                   None when no C++ codegen is needed (pure `@aggregate` use).
    """

    shape        : list[ AxisExpr ]
    ct_variables : list[ str ]      = field( default_factory = list )
    numpy_value  : Optional[ Any ]  = None
    cpp_ref      : Optional[ str ]  = None


class AxisVariableSystem:
    """
    The linear system relating the shapes of a set of tensors to their axis variables,
    for a single scope (the tensors directly contained in one aggregate).

    Each usable axis yields one equation:

        tensor.shape[ axis ] == offset + sum( coeff * value_of( axis_variable ) )

    Axis variables sharing a name across tensors share a column, which is exactly the
    "must be equal" constraint of tensors in the same aggregate. The system is pure:
    it knows nothing about the aggregate tree. Traversal (descending into children,
    ascending to parents for explicit `ctor_kwargs`) is the caller's responsibility.

    `explicit_values` are axis variable values given out-of-band (typically resolved
    `ctor_kwargs`); they take precedence over shape-derived values.
    """

    names    : list[ str ]
    ct_names : list[ str ]

    def __init__( self, sources: list[ AxisTensorSource ], explicit_values: Optional[ dict ] = None ):
        self.sources = list( sources )
        self.explicit_values = dict( explicit_values or {} )

        # axis variable names (run-time + compile-time), in order of appearance
        self.names = []
        self.ct_names = []
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

    # -- queries --------------------------------------------------------------

    def has( self, name: str ) -> bool:
        return name in self.names or name in self.explicit_values

    def _pins( self, num_axis: int, row: list[ int ] ) -> bool:
        """True if `row` determines axis `num_axis` alone (single non-zero coeff)."""
        return row[ num_axis ] != 0 and sum( 1 for c in row if c != 0 ) == 1

    def local_value_of( self, name: str ) -> Optional[ int ]:
        """Concrete value of `name` from explicit values or a pinning tensor shape, else None."""
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

    def check_consistency( self ):
        """Raise if two known tensor shapes imply different values for the same axis variable."""
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

    # -- C++ codegen ----------------------------------------------------------

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
