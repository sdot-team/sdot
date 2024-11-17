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

