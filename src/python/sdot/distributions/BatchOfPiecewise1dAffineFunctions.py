from .BatchOfDistributions import BatchOfDistributions
from ..driver import driver


class BatchOfPiecewise1dAffineFunctions( BatchOfDistributions ):
    """
    xs : tensor[ batch_size, nb_points ]
    ys : tensor[ batch_size, nb_points ]

    if ys is not defined, one uses linspace( self.x0, self.x1, xs.size )
    """

    def __init__( self, xs = None, ys = None, x0 = 0, x1 = 1 ):
        if ys is None and xs is not None:
            xs, ys = ys, xs

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
            return driver.linspace( self.x0, self.x1, self.nb_points )[ None, : ].repeat( ( self.batch_size, 1 ) )
        return self._xs

    @xs.setter
    def xs( self, value ):
        self._xs = driver.t2( value )

    @property
    def ys( self ):
        return self._ys

    @ys.setter
    def ys( self, value ):
        self._ys = driver.t2( value )

    @property
    def batch_size( self ) -> int:
        return self.ys.shape[ 0 ]

    @property
    def nb_points( self ) -> int:
        return self.ys.shape[ 1 ]

    @property
    def dim( self ) -> int:
        return 1

    @property
    def always_1d( self ) -> bool:
        return True
