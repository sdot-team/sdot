from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

from .AxisExpr import AxisExpr


@dataclass
class AxisVariableEquation:
    """
    """

    shape_index : int # -1 means len( shape ). Else, use shape[ shape_index ]

    def _numpy_axis( self ) -> int:
        """First numpy axis occupied by the host expression (sum of the preceding expansions)."""
        return sum( self.shape[ i ].ndim( self.value_of ) for i in range( self.expr_index ) )

    def direct_value( self ):
        """Concrete value (int, or list of ints for an expansion), or None when unavailable."""
        if self.numpy_value is None:
            return None

        if self.kind == "pin":
            return ( int( self.numpy_value.shape[ self._numpy_axis() ] ) - self.offset ) // self.coeff

        if self.kind == "expansion":
            span = self.shape[ self.expr_index ].ndim( self.value_of )
            if not span:
                return None
            axis = self._numpy_axis()
            return [ ( int( self.numpy_value.shape[ axis + i ] ) - self.offset ) // self.coeff for i in range( span ) ]

        if self.kind == "argument":
            nb_with_argument = sum( 1 for e in self.shape if e.has_argument )
            nb_plain_axes = len( self.shape ) - nb_with_argument
            return ( self.numpy_value.ndim - nb_plain_axes - self.offset ) // self.coeff

        return None

