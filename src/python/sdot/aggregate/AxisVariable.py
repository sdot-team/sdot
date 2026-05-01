from __future__ import annotations
from dataclasses import dataclass
# from .AxisExpr import AxisExpr

@dataclass
class AxisVariable:
    """
    A named variable that appears in axis size expressions.

    `arguments`  set by `( ... )` — expanded into one axis per value of each argument.
    `selection`  set by `[ ... ]` — dynamic tensor indexed by these expressions.
    """

    arguments: list | None  # list[ AxisExpr ] ( dim ) — one axis per value of dim
    selection: list | None  # list[ AxisExpr ] [ dim, nb_points ] — dynamic tensor index
    name: str = ''

    def value( self, get_axis_variable, use_dyn_size ):
        name = self.name
        if self.selection is not None:
            if use_dyn_size:
                return get_axis_variable( name, True )
            name = "max_of_" + name
        return get_axis_variable( name, False )

    def unidimensional_version( self ):
        arguments = None
        if self.arguments is not None:
            raise NotImplementedError

        selection = None
        if self.selection is not None:
            selection = []
            for expr in self.selection:
                nexpr = expr.unidimensional_version()
                if nexpr.always_one:
                    continue
                selection.append( nexpr )

        return AxisVariable(
            arguments = arguments,
            selection = selection,
            name = self.name
        )
