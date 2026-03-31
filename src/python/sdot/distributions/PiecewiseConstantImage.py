from .helpers.distribution_methods import TensorField, generate_distribution_methods
from .Distribution import Distribution
from ..driver import driver
import numpy


@generate_distribution_methods
class PiecewiseConstantImage( Distribution ):
    """
    corners : position of min/max points OR position of min + positions of max points along eacg axis
    values : grey values
    """

    corners = TensorField( "nb_corners", "dim" )
    values = TensorField( "*dim" )

    def default_corners( self, is_batch ):
        dim = self.values.ndim - is_batch
        res = numpy.zeros( [ dim + 1, dim ] )
        for d in range( dim ):
            res[ d + 1, d ] = 1
        if is_batch:
            return driver.array( res[ None, :, : ] )
        return driver.array( res )

    def __init__( self, values = None, corners = None ):
        self.corners = corners
        self.values = values
