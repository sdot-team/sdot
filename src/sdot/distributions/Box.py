import numpy

from .Image import Image


def _host_or_traced( a ):
    """`a` as a host array, or as it is when it cannot be read on the host ( traced )"""
    a = getattr( a, "raw", a )
    try:
        return numpy.asarray( a, dtype = float )
    except Exception:
        return a


class Box:
    """A uniform density on a box: `origin + frame^T [ 0, 1 ]^d`. The rows of `frame` are the edges of the box, or `frame`
    is their lengths ( an axis-aligned box ). By default the unit cube, in the dimension of the source it is used with.

    = Internals

    A CONSTANT density on a box -- a parallelepiped -- zero elsewhere.

        rho( x ) = density      if  x = origin + frame^T t  with  t in [ 0, 1 ]^d
                 = 0            otherwise

    The rows of `frame` are the edges of the box ( `frame` can also be a vector of lengths: an axis aligned box ).
    `origin` is `0` and `frame` the identity by default -- the unit cube -- and a default needs the DIMENSION of the space,
    which a `Box()` does not know: it is given where it is known, at the normalization ( `normalized_version( nb_dims )`,
    called by `OtProblem` with the dimension of its source and by `PowerDiagram` with that of its seeds ). Giving `origin` or
    `frame` fixes it earlier, and `nb_dims` says it at once.

    `density`: the value on the box -- meaningless alone, as the distributions are normalized ( to `target_mass` ).

    Once the dimension is known a box IS an `Image` of one pixel -- `normalized_version` returns it -- so that everything an
    image enjoys ( the card's solve, the width continuation, 1D ) is kept. Before that it is only a description, and what is
    asked of it that needs a dimension ( `mass`, `bounding_half_spaces`, ... ) is answered by that image."""

    def __init__( self, origin = None, frame = None, density = 1.0, nb_dims = None, target_mass = 1.0 ):
        self.origin, self.frame, self.density, self._target_mass = origin, frame, density, target_mass
        d = nb_dims
        for a in ( origin, frame ):
            if a is not None:
                shape = tuple( a.shape ) if hasattr( a, "shape" ) else numpy.asarray( a ).shape
                if d is not None and int( d ) != int( shape[ 0 ] ):
                    raise ValueError( f"Box : `nb_dims` = { d } but a parameter is of shape { shape }" )
                d = shape[ 0 ]
        self._nb_dims = None if d is None else int( d )
        self._images = {}

    @property
    def nb_dims_or_none( self ):
        """= Internals

        the dimension, or `None` while the box has no default to make up"""
        return self._nb_dims

    def image( self, nb_dims = None ):
        """= Internals

        the `Image` of one pixel that this box is, in dimension `nb_dims` ( or its own )"""
        d = self._nb_dims if self._nb_dims is not None else nb_dims
        if d is None:
            raise ValueError( "Box : the dimension of the space is not known yet ( give `origin`, `frame` or `nb_dims`, "
                              "or use the box where it is known: an `OtProblem`, a `PowerDiagram` )" )
        if self._nb_dims is not None and nb_dims is not None and int( nb_dims ) != self._nb_dims:
            raise ValueError( f"Box : a box of dimension { self._nb_dims } used in dimension { nb_dims }" )
        d = int( d )
        if d not in self._images:
            # ( a traced origin stays traced: the plan differentiates it by translation, `SdotPlanNd._origin_as_shift` )
            origin = numpy.zeros( d ) if self.origin is None else _host_or_traced( self.origin ).reshape( d )
            if self.frame is None:
                frame = numpy.eye( d )
            else:
                frame = _host_or_traced( self.frame )               # ( traced: the integration derives it, `diagram::AffineAcc` )
                if frame.ndim == 1:
                    frame = numpy.diag( frame ) if isinstance( frame, numpy.ndarray ) else frame[ :, None ] * numpy.eye( d )
                else:
                    frame = frame.reshape( d, d )
            # ( a traced density stays traced: the plan is differentiable with respect to it -- with a zero derivative, the
            # distribution being normalized -- rather than refusing it )
            dens = getattr( self.density, "raw", self.density )
            values = numpy.full( ( 1, ) * d, float( dens ) ) if isinstance( dens, ( int, float, numpy.number ) ) else numpy.ones( ( 1, ) * d ) * dens
            self._images[ d ] = Image( values = values, origin = origin, pixel_frame = frame,
                                     target_mass = self._target_mass )
        return self._images[ d ]

    def normalized_version( self, nb_dims = None ):
        return self.image( nb_dims ).normalized_version()

    def __getattr__( self, name ):
        # what only an image can answer ( `mass`, `bounding_half_spaces`, `nb_dims`, ... ), once the dimension is known
        if name.startswith( "_" ):
            raise AttributeError( name )
        return getattr( self.image(), name )
