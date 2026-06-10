from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CppScalar:
    """
    A C++ integer expression taking part in axis-variable resolution.

    `AxisExpr.solve` combines tensor shapes using only `-`, `*`, `//`; with `CppScalar`
    operands these build a C++ expression instead of computing an int, so the very same
    resolution code yields the C++ value of an axis variable (its `str()` is the code).
    Operands may be ints or `CppScalar`s; the identities `-0`, `*1`, `*0`, `/1` are folded
    to keep the generated code readable.
    """

    code : str

    def __str__( self ) -> str:
        return self.code

    def __sub__( self, other ) -> CppScalar:
        if isinstance( other, int ) and other == 0:
            return self
        return CppScalar( f"( { self.code } - { other } )" )

    def __mul__( self, other ) -> CppScalar:
        if isinstance( other, int ):
            if other == 1:
                return self
            if other == 0:
                return CppScalar( "0" )
        return CppScalar( f"{ self.code } * { other }" )

    __rmul__ = __mul__

    def __floordiv__( self, other ) -> CppScalar:
        if isinstance( other, int ) and other == 1:
            return self
        return CppScalar( f"( { self.code } ) / { other }" )
