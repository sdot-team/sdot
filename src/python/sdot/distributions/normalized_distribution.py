from ..space_subsets.SpaceSubset import SpaceSubset
from .IndicatorFunction import IndicatorFunction
from .ConstantValue import ConstantValue
from .Distribution import Distribution

def normalized_distribution( dist, return_one_if_not_defined = True ):
    """ return a Distribution instance from dist, which can be a scalar, a SpaceSubset, ... """

    if dist is None:
        if return_one_if_not_defined:
            return ConstantValue( 1 )
        return None
    
    if isinstance( dist, Distribution ):
        return dist 
    
    if isinstance( dist, float ) or isinstance( dist, int ):
        return ConstantValue( dist )
    
    if isinstance( dist, SpaceSubset ):
        return IndicatorFunction( dist )
    
    raise ValueError( f"don't know how to make a distribution from a { type( dist ) }" )
