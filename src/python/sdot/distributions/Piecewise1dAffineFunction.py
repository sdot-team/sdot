from .BatchOfPiecewise1dAffineFunctions import BatchOfPiecewise1dAffineFunctions
from .Distribution import Distribution
from ..driver import driver


class Piecewise1dAffineFunction( Distribution ):
    """
    xs : tensor[ nb_points ]
    ys : tensor[ nb_points ]

    if ys is not defined, one uses linspace( self.x0, self.x1, xs.size )
    """
    def __init__( self, xs = None, ys = None, x0 = 0, x1 = 1 ):
        """

        x0 and x1 are used only if xs is not specified
        """
        self._xs = None
        self._ys = None
        self.x0 = x0
        self.x1 = x1

        self.xs = xs
        self.ys = ys

    @property
    def xs( self ):
        if self._xs is None:
            ys = self.ys
            if ys is None:
                return None
            return driver.linspace( self.x0, self.x1, ys.shape[ -1 ] )
        return self._xs

    @xs.setter
    def xs( self, value ):
        self._xs = driver.t1( value )

    @property
    def ys( self ):
        return self._ys

    @ys.setter
    def ys( self, value ):
        self._ys = driver.t1( value )

    @property
    def batch_size( self ):
        return self.ys.shape[ 0 ]

    def batch_version(self):
        x = self.xs
        if x is not None:
            x = x[ None, : ]
        y = self.ys
        if y is not None:
            y = y[ None, : ]
        return BatchOfPiecewise1dAffineFunctions( x, y )
