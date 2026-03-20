from .Distribution import Distribution
from sdot.distributions.distribution_methods import TensorField, generate_distribution_methods
from ..driver import driver


@generate_distribution_methods
class PiecewiseAffineFunction1d( Distribution ):
    """
    xs : tensor[ nb_points ]   -- optional: uses linspace( x0, x1, nb_points ) if not set
    ys : tensor[ nb_points ]

    x0, x1 : interval bounds (used only when xs is not provided)
    """

    x0 = 0
    x1 = 1

    xs = TensorField( "nb_points" )
    ys = TensorField( "nb_points" )

    def default_xs( self ):
        return driver.linspace( self.x0, self.x1, self.ys.shape[ -1 ] ) if self.ys is not None else None

    def __init__( self, xs = None, ys = None, x0 = 0, x1 = 1 ):
        # Convenience: single positional argument is treated as ys
        if ys is None and xs is not None:
            xs, ys = ys, xs
        self.x0 = x0
        self.x1 = x1
        self.xs = xs
        self.ys = ys

    # declared for IDE autocompletion — generated at runtime from axis names
    nb_points: int
