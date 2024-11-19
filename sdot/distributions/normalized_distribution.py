from .SymbolicFunction import SymbolicFunction
from .Distribution import Distribution
from .UnitBox import UnitBox

def normalized_distribution( dist, return_unit_box_if_not_defined = True ):
    """ return a Distribution instance from `dist`, which can be a scalar, a string, ... """

    # default value
    if dist is None:
        if return_unit_box_if_not_defined:
            return UnitBox()
        return None
    
    # already the good type
    if isinstance( dist, Distribution ):
        return dist 

    # scalar    
    if isinstance( dist, float ) or isinstance( dist, int ) or isinstance( dist, str ):
        return SymbolicFunction( dist )

    # not found    
    raise ValueError( f"don't know how to make a distribution from a { type( dist ) }" )
