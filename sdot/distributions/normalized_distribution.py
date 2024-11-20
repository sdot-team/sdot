from .SymbolicFunction import SymbolicFunction
from .Distribution import Distribution
from .Lebesgue import Lebesgue
from .UnitBox import UnitBox

def normalized_distribution( dist, if_not_defined = "UnitBox" ):
    """ return a Distribution instance from `dist`, which can be a scalar, a string, ... """

    # default value
    if dist is None:
        if if_not_defined == "Lebesgue":
            return Lebesgue()
        if if_not_defined == "UnitBox":
            return UnitBox()
        raise ValueError( f"Undefined distribution name { if_not_defined }" )
    
    # already the good type
    if isinstance( dist, Distribution ):
        return dist 

    # scalar    
    if isinstance( dist, float ) or isinstance( dist, int ) or isinstance( dist, str ):
        return SymbolicFunction( dist )

    # not found    
    raise ValueError( f"don't know how to make a distribution from a { type( dist ) }" )
