import numpy

from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar
from loom.util import ComputedAttribute

from ._bvh import build_bvh
from .Distribution import Distribution


class CodimMesh( Distribution ):
    """A measure carried by a mesh M of LOWER dimension: segments in 2D or 3D, triangles in 3D ( `Mesh( nodes, simplices )` with
    fewer than `d + 1` nodes per simplex gives one ). `values` is `rho` at the nodes of M ( P1, ones by default ): the measure is
    `rho` times the length ( area ) on M, and its cells are integrated exactly ( `CodimMeasureDensity` ).

    THE WIDTH CONTINUATION of the solver goes through smoothed densities, `rho exp( -d^2 / s^2 )`, `d` the distance to M and `rho` its
    value at the closest point of M, integrated in closed form on the ENVELOPE of M ( `sdot.envelope`: the generalized Voronoi
    diagram of its elements, simplices where `d` and `rho` are affine, in the box of M widened by `margin` times its diagonal ), from
    wide down to `width`, and then the measure itself ( `sdotplan/convolved/CodimMesh.h` ). Default `width`: `DEFAULT_WIDTH` times
    the diagonal of M. `smoothed = True` gives the smoothed density at `width` instead of the measure ( a check, or a
    density of its own ).

    `alpha`: the widest angle of the fans of the envelope, in degrees ( `d` is over-estimated by at most `1 / cos( alpha / 2 )`
    around the nodes, and the edges in 3D ). `quadrature = True` integrates the smoothed pieces by `PointwiseDensity` ( a check ).

    = Internals

    The envelope is stored as a soup: every simplex has its own corners ( `nodes` = the corners, `simplices` their indices ),
    `values` is `w rho` and `dists` `d` at the corners ( `w` the weight of the fans around lines in 3D ). The rest is as `Mesh`:
    the gradients of the barycentric coordinates, the hierarchy of boxes. M itself is `m_nodes`, `m_simplices`, `m_rho`, with its
    own hierarchy. `params = [ width, quadrature, smoothed ]`.
    """

    nb_nodes         : ShapeVar
    nb_elems         : ShapeVar
    nb_corners       : ShapeVar
    nb_bnodes        : ShapeVar
    nb_links         : ShapeVar
    nb_params        : ShapeVar
    nb_mnodes        : ShapeVar
    nb_melems        : ShapeVar
    nb_mverts        : ShapeVar
    nb_mbnodes       : ShapeVar
    nb_dims          : CtShapeVar

    num_node         : Axis[ "nb_nodes" ]
    num_elem         : Axis[ "nb_elems" ]
    num_corner       : Axis[ "nb_corners" ]
    num_bnode        : Axis[ "nb_bnodes" ]
    num_link         : Axis[ "nb_links" ]
    num_param        : Axis[ "nb_params" ]
    num_mnode        : Axis[ "nb_mnodes" ]
    num_melem        : Axis[ "nb_melems" ]
    num_mvert        : Axis[ "nb_mverts" ]
    num_mbnode       : Axis[ "nb_mbnodes" ]
    dim              : Axis[ "nb_dims" ]

    nodes            : RealTensor[ "num_node", "dim" ]
    values           : RealTensor[ "num_elem", "num_corner" ]                 # `w rho` at the corners
    dists            : RealTensor[ "num_elem", "num_corner" ]                 # `d` at the corners
    simplices        : IntTensor[ "num_elem", "num_corner", dict( size = 32 ) ]
    grads            : RealTensor[ "num_elem", "num_corner", "dim" ]
    bvh_lo           : RealTensor[ "num_bnode", "dim" ]
    bvh_hi           : RealTensor[ "num_bnode", "dim" ]
    bvh_links        : IntTensor[ "num_bnode", "num_link", dict( size = 32 ) ]
    params           : RealTensor[ "num_param" ]                             # width, quadrature, smoothed
    m_nodes          : RealTensor[ "num_mnode", "dim" ]
    m_rho            : RealTensor[ "num_mnode" ]
    m_simplices      : IntTensor[ "num_melem", "num_mvert", dict( size = 32 ) ]
    m_bvh_lo         : RealTensor[ "num_mbnode", "dim" ]
    m_bvh_hi         : RealTensor[ "num_mbnode", "dim" ]
    m_bvh_links      : IntTensor[ "num_mbnode", "num_link", dict( size = 32 ) ]

    current_mass     : ComputedAttribute[ RealTensor, ( "m_rho", ) ]

    cuts_pieces = True

    #: the default `width`, times the diagonal of M: the last smoothed stage must be thin enough for every cell to cross M, the
    #: measure has no mass elsewhere ( at `1e-3`, cells of the smoothed solution still lie in layers along M without crossing it )
    DEFAULT_WIDTH = 1e-5

    def __init__( self, nodes, simplices, values = None, width = None, target_mass = 1.0, alpha = 15.0, margin = 0.25,
                  quadrature = False, smoothed = False, **kwargs ):
        """see the class docstring. The other keyword arguments are for the copies ( the envelope already built )"""
        if "grads" in kwargs:
            self._box = kwargs.pop( "box" )
            self._m_box = kwargs.pop( "m_box" )
            self.__base_init__( nodes = nodes, simplices = simplices, values = values, target_mass = target_mass, **kwargs )
            return

        from ..envelope import envelope
        nd = numpy.asarray( nodes, dtype = float )
        sx = numpy.asarray( simplices )
        if nd.ndim != 2 or sx.ndim != 2 or ( nd.shape[ 1 ], sx.shape[ 1 ] ) not in ( ( 2, 2 ), ( 3, 2 ), ( 3, 3 ) ):
            raise ValueError( f"CodimMesh: segments in 2D or 3D, or triangles in 3D ( got `nodes` { nd.shape }, `simplices` { sx.shape } )" )
        d = nd.shape[ 1 ]
        rho = numpy.ones( len( nd ) ) if values is None else numpy.asarray( values, dtype = float )
        a, b = nd.min( 0 ), nd.max( 0 )
        if width is None:
            width = self.DEFAULT_WIDTH * float( numpy.linalg.norm( b - a ) )
        m = margin * numpy.linalg.norm( b - a )
        lo, hi = a - m, b + m
        env = envelope( nd, sx, rho, lo = lo, hi = hi, alpha = alpha )

        P = env[ "pts" ]
        A = P[ :, 1:, : ] - P[ :, :1, : ]
        keep = numpy.abs( numpy.linalg.det( A ) ) > 1e-14 * float( numpy.linalg.norm( hi - lo ) ) ** d
        P, A = P[ keep ], A[ keep ]
        G = numpy.empty( ( len( P ), d + 1, d ) )
        G[ :, 1:, : ] = numpy.swapaxes( numpy.linalg.inv( A ), 1, 2 )
        G[ :, 0, : ] = - G[ :, 1:, : ].sum( axis = 1 )
        perm, bvh_lo, bvh_hi, links = build_bvh( P.min( axis = 1 ), P.max( axis = 1 ) )
        P, G = P[ perm ], G[ perm ]
        cv = ( env[ "rho" ] * env[ "w" ][ :, None ] )[ keep ][ perm ]
        dv = env[ "d" ][ keep ][ perm ]
        Q = nd[ sx ]                                            # M, in the order of its own hierarchy
        mperm, m_lo, m_hi, m_links = build_bvh( Q.min( axis = 1 ), Q.max( axis = 1 ) )
        self._box = ( lo, hi )
        self._m_box = ( a, b )
        self.__base_init__( nodes = P.reshape( -1, d ), simplices = numpy.arange( P.shape[ 0 ] * ( d + 1 ), dtype = numpy.int32 ).reshape( -1, d + 1 ),
                            values = cv, dists = dv, grads = G, bvh_lo = bvh_lo, bvh_hi = bvh_hi, bvh_links = links.astype( numpy.int32 ),
                            params = numpy.array( [ float( width ), float( bool( quadrature ) ), float( bool( smoothed ) ) ] ),
                            m_nodes = nd, m_rho = rho, m_simplices = sx[ mperm ].astype( numpy.int32 ), m_bvh_lo = m_lo, m_bvh_hi = m_hi,
                            m_bvh_links = m_links.astype( numpy.int32 ), target_mass = target_mass )

    def _copy( self, values = None, m_rho = None, params = None, current_mass = None ):
        return CodimMesh(
            self.nodes, self.simplices, values = self.values if values is None else values, target_mass = self.target_mass,
            dists = self.dists, grads = self.grads, bvh_lo = self.bvh_lo, bvh_hi = self.bvh_hi, bvh_links = self.bvh_links,
            params = self.params if params is None else params,
            m_nodes = self.m_nodes, m_rho = self.m_rho if m_rho is None else m_rho, m_simplices = self.m_simplices,
            m_bvh_lo = self.m_bvh_lo, m_bvh_hi = self.m_bvh_hi, m_bvh_links = self.m_bvh_links,
            batch_axes = self.batch_axes, box = self._box, m_box = self._m_box, **( {} if current_mass is None else dict( current_mass = current_mass ) ),
        )

    @property
    def width( self ):
        return float( numpy.asarray( self.params )[ 0 ] )

    @property
    def smoothed( self ):
        return bool( numpy.asarray( self.params )[ 2 ] )

    def with_width( self, width, quadrature = None, smoothed = None ):
        """the same envelope at another width ( and `quadrature`, `smoothed` )"""
        p = numpy.asarray( self.params, dtype = float ).copy()
        p[ 0 ] = width
        if quadrature is not None:
            p[ 1 ] = float( bool( quadrature ) )
        if smoothed is not None:
            p[ 2 ] = float( bool( smoothed ) )
        return self._copy( params = p )

    def normalized_version( self, nb_dims = None ):
        """`rho` and the envelope scaled together, for the measure ( or the smoothed density ) to have the target mass"""
        mass = self.mass
        if not self.target_mass.is_defined:
            return self
        f = self.target_mass / mass
        return self._copy( values = f * self.values, m_rho = f * self.m_rho, current_mass = self.target_mass )

    def m_integrals( self ):
        """`int rho` over each simplex of M ( host arrays )"""
        nd = numpy.asarray( self.m_nodes, dtype = float )
        sx = numpy.asarray( self.m_simplices )
        r = numpy.asarray( self.m_rho, dtype = float )[ sx ].mean( axis = 1 )
        Q = nd[ sx ]
        if sx.shape[ 1 ] == 2:
            return numpy.linalg.norm( Q[ :, 1 ] - Q[ :, 0 ], axis = 1 ) * r
        return 0.5 * numpy.linalg.norm( numpy.cross( Q[ :, 1 ] - Q[ :, 0 ], Q[ :, 2 ] - Q[ :, 0 ] ), axis = 1 ) * r

    def random_points( self, n, seed = None ):
        """`n` points drawn on M ( a host `[ n, d ]` array ): the simplices by their mass, then uniformly in each one -- the start
        of `sdot.quantization`, admissible for the measure on M itself ( drawn in the box, most seeds miss M and the first solve
        goes through the whole continuation )"""
        rng = numpy.random.default_rng( seed )
        mass = numpy.maximum( self.m_integrals(), 0 )
        Q = numpy.asarray( self.m_nodes, dtype = float )[ numpy.asarray( self.m_simplices ) ]
        e = rng.choice( len( Q ), n, p = mass / mass.sum() )
        b = rng.exponential( size = ( n, Q.shape[ 1 ] ) )                 # ( uniform barycentric coordinates )
        b /= b.sum( axis = 1, keepdims = True )
        return numpy.einsum( "ni,nic->nc", b, Q[ e ] )

    def element_integrals( self ):
        """`int rho exp( -d^2 / width^2 )` over each simplex of the envelope ( host arrays, closed forms ): the smoothed density"""
        import loom
        from loom.compilation.FfiCode import FfiCode
        nd = numpy.asarray( self.nodes, dtype = float )
        d = nd.shape[ 1 ]
        sx = numpy.asarray( self.simplices )
        P = numpy.ascontiguousarray( nd[ sx ] )
        num_elem = Axis( ShapeVar( len( sx ) ), name = "num_elem" )
        num_corner = Axis( ShapeVar( d + 1 ), name = "num_corner" )
        dim = Axis( ShapeVar( d ), name = "dim" )
        out = RealTensor[ num_elem ]()
        loom.ffi_call(
            "codim_mesh_integrals",
            FfiCode.inline( f"sdot::codim::element_integrals<{ d }>( queue, args.inputs.pts, args.inputs.dists, args.inputs.rhos, args.inputs.params, args.outputs.out );",
                            includes = [ "sdot/integration/CodimMeshIntegrals.h" ] ),
            pts = RealTensor[ num_elem, num_corner, dim ]( P ),
            dists = RealTensor[ num_elem, num_corner ]( numpy.ascontiguousarray( self.dists, dtype = float ) ),
            rhos = RealTensor[ num_elem, num_corner ]( numpy.ascontiguousarray( self.values, dtype = float ) ),
            params = RealTensor[ Axis( ShapeVar( len( numpy.asarray( self.params ) ) ), name = "num_param" ) ]( numpy.asarray( self.params, dtype = float ) ),
            out = loom.out( out ),
            has_dynamic_capacity = False,
        )
        return numpy.asarray( out ).reshape( -1 )

    def _update_current_mass( self ):
        self.current_mass = float( self.element_integrals().sum() if self.smoothed else self.m_integrals().sum() )

    def bounding_half_spaces( self ):
        """the box of the envelope: the density is positive in all of it"""
        lo, hi = self._box
        d = len( lo )
        return numpy.concatenate( [ numpy.eye( d ), -numpy.eye( d ) ] ), numpy.concatenate( [ hi, -lo ] )

    def domain_box( self ):
        return self._box

    def inscribed_box( self ):
        """the box of M, widened by the width ( flat for a straight line ): the density is positive everywhere, and largest there"""
        a, b = self._m_box
        pad = max( self.width, 1e-3 * float( numpy.linalg.norm( b - a ) ) )
        return a - pad, b + pad
