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
