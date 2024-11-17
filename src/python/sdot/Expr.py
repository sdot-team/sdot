from .bindings.loader import module_for
# from types import ModuleType
import numpy as np

class Expr:
    """ wrapper around cpp sdot::Expr class to store symbolic expressions """

    def __init__( self, value = None ):
        """ 
        """
        self._module = module_for( 'generic_objects' )
        self._expr = self._module.Expr( value )

    def __repr__( self ):
        return self._expr.__repr__()

    def __add__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return self._expr.add( that._expr )

    def __sub__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return self._expr.sub( that._expr )

    def __mul__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return self._expr.mul( that._expr )

    def __div__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return self._expr.div( that._expr )
    
    def __pow__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return self._expr.pow( that._expr )
    
