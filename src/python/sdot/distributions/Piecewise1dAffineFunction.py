from .Distribution import Distribution
from sdot.distributions.distribution_methods import TensorField
from ..driver import driver


class Piecewise1dAffineFunction( Distribution ):
    """
    xs : tensor[ nb_points ]   — optionnel : linspace( x0, x1, nb_points ) si absent
    ys : tensor[ nb_points ]

    x0, x1 : bornes de l'intervalle (utilisées uniquement si xs n'est pas fourni)
    """

    x0 = 0
    x1 = 1

    xs = TensorField( 1, default = lambda self:
        driver.linspace( self.x0, self.x1, self.ys.shape[ -1 ] ) if self.ys is not None else None
    )
    ys = TensorField( 1 )

    def __init__( self, xs = None, ys = None, x0 = 0, x1 = 1 ):
        # Commodité : un seul argument positionnel → traité comme ys
        if ys is None and xs is not None:
            xs, ys = ys, xs
        self.x0 = x0
        self.x1 = x1
        self.xs = xs
        self.ys = ys

    @property
    def dim( self ):
        return 1

    @property
    def always_1d( self ):
        return True

    # batch_version → généré : BatchOfPiecewise1dAffineFunction
