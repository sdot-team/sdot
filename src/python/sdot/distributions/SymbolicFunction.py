from .Distribution import Distribution
from ..Expr import Expr

class SymbolicFunction( Distribution ):
    """ 
    """

    def __init__( self, expr ):
        """ 
        """
        self.expr = Expr( expr )

    def __repr__( self ):
        return self.expr.__repr__()

    def boundary_split( self, ndim = None ):
        return self.expr.boundary_split( ndim )
