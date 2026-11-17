"""The domains of pysdot ( `pysdot.domain_types` ): densities that are converted to an sdot distribution when a diagram uses them.

`ConvexPolyhedraAssembly` becomes:

- a `sdot.Box` for one box;
- an `sdot.Image` on the grid of all the coordinates of the boxes for several ( axis-aligned ) boxes -- what makes `sub_box` and
  overlapping boxes exact: each pixel gets the sum of the coefficients of the boxes that cover it;
- an `sdot.Polytope` for one simplex or one convex polyhedron;
- an `sdot.Mesh` ( constant on each simplex ) for several simplices and boxes, which must then not overlap.

`ScaledImage( min_pt, max_pt, img )` becomes an `sdot.Image` ( `img[ y, x ]` in 2D, as in pysdot ).

The integrals of pysdot are not normalized: `measure()` is the mass of the density, and the diagrams multiply what sdot returns
( normalized to 1 ) by it.
"""
import numpy as np
import math


class ConvexPolyhedraAssembly:
    def __init__( self, type = np.float64 ):
        self._type = type
        self._pieces = []          # ( kind, data, coeff ): "box" ( mi, ma ), "simplex" points, "polyhedron" ( directions, offsets )
        self._cache = None
        self.dimension = None

    # ---- construction ---------------------------------------------------------------------------------------
    def add_box( self, min_pos, max_pos, coeff = 1.0, cut_id = -1 ):
        mi, ma = np.asarray( min_pos, dtype = float ).reshape( -1 ), np.asarray( max_pos, dtype = float ).reshape( -1 )
        self._add( "box", ( np.minimum( mi, ma ), np.maximum( mi, ma ) ), coeff, mi.size )

    def sub_box( self, min_pos, max_pos, coeff = 1.0, cut_id = -1 ):
        self.add_box( min_pos, max_pos, - coeff, cut_id )

    def add_simplex( self, points, coeff = 1.0, cut_id = -1 ):
        pts = np.asarray( points, dtype = float )
        self._add( "simplex", pts, coeff, pts.shape[ 1 ] )

    def add_convex_polyhedron( self, positions_and_normals, coeff = 1.0, cut_id = -1 ):
        """`positions_and_normals`: one row `[ p..., n... ]` per face, a point of the face and its OUTWARD normal"""
        pan = np.asarray( positions_and_normals, dtype = float )
        d = pan.shape[ 1 ] // 2
        p, n = pan[ :, :d ], pan[ :, d: ]
        self._add( "polyhedron", ( n, ( n * p ).sum( axis = 1 ) ), coeff, d )

    def add_img( self, min_pos, max_pos, img, cut_id = -1 ):
        """one box per pixel ( `img[ y, x ]` ) -- `ScaledImage` does the same in one piece"""
        img = np.asarray( img, dtype = float )
        sx = 1.0 * ( max_pos[ 0 ] - min_pos[ 0 ] ) / img.shape[ 1 ]
        sy = 1.0 * ( max_pos[ 1 ] - min_pos[ 1 ] ) / img.shape[ 0 ]
        for y in range( img.shape[ 0 ] ):
            for x in range( img.shape[ 1 ] ):
                self.add_box( [ min_pos[ 0 ] + sx * x, min_pos[ 1 ] + sy * y ], [ min_pos[ 0 ] + sx * ( x + 1 ), min_pos[ 1 ] + sy * ( y + 1 ) ], img[ y, x ] )

    def _add( self, kind, data, coeff, d ):
        self._arrangement = None
        if self.dimension is not None and d != self.dimension:
            raise ValueError( f"ConvexPolyhedraAssembly: a piece in { d }D in an assembly in { self.dimension }D" )
        self.dimension = d
        self._pieces.append( ( kind, data, float( coeff ) ) )
        self._cache = None

    # ---- what is read ---------------------------------------------------------------------------------------
    def measure( self ):
        """the integral of the density ( the sum of `coeff * volume` over the pieces )"""
        return self._type( sum( c * _piece_volume( k, x ) for k, x, c in self._pieces ) )

    def coeff_at( self, point ):
        """the density at `point` ( 0 outside )"""
        x = np.asarray( point, dtype = float ).reshape( -1 )
        return self._type( sum( c for k, data, c in self._pieces if _piece_contains( k, data, x ) ) )

    def contains( self, point ):
        return self.coeff_at( point ) != 0

    def min_position( self ):
        return np.min( [ _piece_box( k, x )[ 0 ] for k, x, c in self._pieces ], axis = 0 )

    def max_position( self ):
        return np.max( [ _piece_box( k, x )[ 1 ] for k, x, c in self._pieces ], axis = 0 )

    def display_boundaries_vtk( self, filename ):
        from .PowerDiagram import _write_vtk_polygons
        polys = []
        for k, data, c in self._pieces:
            polys += _piece_faces( k, data )
        _write_vtk_polygons( filename, polys, { "coeff": [ c for k, data, c in self._pieces for _ in _piece_faces( k, data ) ] } )

    def pieces_half_spaces( self ):
        """`[ ( directions, offsets, coeff ) ]`: convex pieces where the density is positive, as half-spaces ( what the outputs clip
        the cells with, as pysdot did ) -- the pieces themselves when they are all positive, the arrangement in 2D otherwise"""
        if all( c > 0 for _, _, c in self._pieces ) or self.dimension != 2:
            return [ _piece_half_spaces( k, data ) + ( c, ) for k, data, c in self._pieces if c > 0 ]
        return [ _polygon_half_spaces( poly ) + ( c, ) for poly, c in self.arrangement_2d() if c > 0 ]

    def arrangement_2d( self ):
        """`[ ( polygon, coeff ) ]`: disjoint convex polygons ( counter clockwise ) covering the pieces, each with the sum of the
        coefficients of the pieces over it -- what `sub_box` and the holes ( negative coefficients ) mean. Zero ones left out."""
        if getattr( self, "_arrangement", None ) is None:
            from .PowerDiagram import _clip_polygon
            cells = []                                       # ( polygon, coeff ), disjoint
            for k, data, c in self._pieces:
                P = _piece_polygon( k, data )
                dirs, offs = _polygon_half_spaces( P )
                lo, hi = P.min( axis = 0 ), P.max( axis = 0 )
                new, rest = [], [ P ]
                for Q, q in cells:
                    if np.any( Q.max( axis = 0 ) < lo ) or np.any( Q.min( axis = 0 ) > hi ):
                        new.append( ( Q, q ) )
                        continue
                    inter = _clip_polygon( Q, dirs, offs )
                    if _area( inter ) <= 0:
                        new.append( ( Q, q ) )
                        continue
                    new.append( ( inter, q + c ) )
                    new += [ ( R, q ) for R in _difference( Q, dirs, offs ) ]
                    qd, qo = _polygon_half_spaces( Q )
                    rest = [ R2 for R in rest for R2 in _difference( R, qd, qo ) ]
                cells = new + [ ( R, c ) for R in rest ]
            if any( c < -1e-12 for _, c in cells ):
                raise ValueError( "ConvexPolyhedraAssembly: the pieces give a negative density somewhere" )
            self._arrangement = [ ( p, c ) for p, c in cells if abs( c ) > 1e-12 and _area( p ) > 0 ]
        return self._arrangement

    # ---- to sdot --------------------------------------------------------------------------------------------
    def to_sdot( self, target_mass = 1.0 ):
        """the sdot distribution of this density, normalized to `target_mass` ( `measure()` gives its own mass )"""
        if self._cache is None:
            self._cache = {}
        if target_mass not in self._cache:
            self._cache[ target_mass ] = self._make_sdot( target_mass )
        return self._cache[ target_mass ]

    def _make_sdot( self, tm ):
        import sdot
        if not self._pieces:
            raise ValueError( "ConvexPolyhedraAssembly: empty assembly" )
        d = self.dimension
        kinds = { k for k, _, _ in self._pieces }
        if kinds == { "box" }:
            if len( self._pieces ) == 1:
                ( mi, ma ), c = self._pieces[ 0 ][ 1 ], self._pieces[ 0 ][ 2 ]
                return sdot.Box( origin = mi, frame = ma - mi, density = c, target_mass = tm )
            return _boxes_as_image( self._pieces, d, tm )
        if len( self._pieces ) == 1:
            k, data, c = self._pieces[ 0 ]
            if k == "simplex":
                return sdot.ConvexHull( data, density = c, target_mass = tm )
            return sdot.Polytope( data[ 0 ], data[ 1 ], density = c, target_mass = tm )
        if d == 2:
            # any pieces, overlapping or not: the arrangement ( disjoint convex polygons, each the sum of the coefficients over it ),
            # cut in triangles -- a mesh constant on each one
            nodes, simplices, values = [], [], []
            for poly, c in self.arrangement_2d():
                for k in range( 1, len( poly ) - 1 ):
                    simplices.append( np.arange( len( nodes ), len( nodes ) + 3 ) )
                    nodes += [ poly[ 0 ], poly[ k ], poly[ k + 1 ] ]
                    values.append( [ c ] * 3 )
            return sdot.Mesh( np.array( nodes ), np.array( simplices ), np.array( values, dtype = float ), degree = 0, target_mass = tm )
        if "polyhedron" in kinds:
            raise NotImplementedError( "sdot.pysdot: in 3D, an assembly with several pieces among which a convex polyhedron -- "
                                       "boxes alone, simplices and boxes, or a single polyhedron are supported" )
        # simplices ( and boxes, cut in simplices ): a mesh constant on each simplex. They must not overlap
        nodes, simplices, values = [], [], []
        for k, data, c in self._pieces:
            for s in ( [ data ] if k == "simplex" else _box_simplices( *data ) ):
                simplices.append( np.arange( len( nodes ), len( nodes ) + d + 1 ) )
                nodes += list( s )
                values.append( [ c ] * ( d + 1 ) )
        return sdot.Mesh( np.array( nodes ), np.array( simplices ), np.array( values, dtype = float ), degree = 0, target_mass = tm )


class ScaledImage:
    """a density constant on each pixel of `img` ( `img[ y, x ]` in 2D, `img[ z, y, x ]` in 3D ), on the box `[ min_pt, max_pt ]`.
    In 2D, `img[ c, y, x ]` with 6 coefficients per pixel: the polynomial `c0 + c1 x + c2 y + c3 x^2 + c4 x y + c5 y^2` ( global
    coordinates ), as in pysdot -- an `sdot.Image( degree = 2 )`."""
    def __init__( self, min_pt, max_pt, img ):
        self._type = np.float64
        self._min = np.asarray( min_pt, dtype = float ).reshape( -1 )
        self._max = np.asarray( max_pt, dtype = float ).reshape( -1 )
        img = np.ascontiguousarray( img, dtype = float )
        self.dimension = self._min.size
        self._poly = None
        if img.ndim == self.dimension + 1:
            if img.shape[ 0 ] == 1:
                img = img[ 0 ]
            elif self.dimension == 2 and img.shape[ 0 ] == 6:
                self._poly, img = img, img[ 0 ]
            else:
                raise NotImplementedError( f"ScaledImage: { img.shape[ 0 ] } coefficients per pixel in { self.dimension }D ( 1, or 6 in 2D )" )
        if img.ndim != self.dimension:
            raise ValueError( f"ScaledImage: a { img.ndim }D image in a { self.dimension }D box" )
        self._img = img
        self._cache = None

    def _bernstein( self ):
        """`[ nx, ny, 9 ]`: the coefficients of the polynomial pixels in the Bernstein basis of degree 2 of each pixel ( x the slowest )"""
        ny, nx = self._img.shape
        h = ( self._max - self._min ) / np.array( [ nx, ny ] )
        u = np.array( [ 0.0, 0.5, 1.0 ] )
        C = np.stack( [ ( 1 - u ) ** 2, 2 * u * ( 1 - u ), u ** 2 ], axis = 1 )      # C[ node, basis ]
        K = np.kron( C, C )
        ix, iy = np.meshgrid( np.arange( nx ), np.arange( ny ), indexing = "ij" )
        t, s = np.meshgrid( u, u, indexing = "ij" )
        x = self._min[ 0 ] + h[ 0 ] * ( ix[ :, :, None ] + t.reshape( -1 )[ None, None, : ] )
        y = self._min[ 1 ] + h[ 1 ] * ( iy[ :, :, None ] + s.reshape( -1 )[ None, None, : ] )
        c = [ self._poly[ k ].T[ :, :, None ] for k in range( 6 ) ]
        vals = c[ 0 ] + c[ 1 ] * x + c[ 2 ] * y + c[ 3 ] * x * x + c[ 4 ] * x * y + c[ 5 ] * y * y
        return np.linalg.solve( K, vals.reshape( -1, 9 ).T ).T.reshape( nx, ny, 9 )

    def measure( self ):
        pix = float( np.prod( ( self._max - self._min ) / np.array( self._img.shape[ ::-1 ] ) ) )
        if self._poly is not None:
            return float( self._bernstein().mean( axis = 2 ).sum() * pix )
        return float( self._img.sum() * pix )

    def coeff_at( self, point ):
        x = np.asarray( point, dtype = float ).reshape( -1 )
        if np.any( x < self._min ) or np.any( x >= self._max ):
            return 0.0
        sh = np.array( self._img.shape[ ::-1 ] )
        i = tuple( np.minimum( ( ( x - self._min ) / ( self._max - self._min ) * sh ).astype( int ), sh - 1 )[ ::-1 ] )
        if self._poly is not None:
            c = self._poly[ ( slice( None ), ) + i ]
            return float( c[ 0 ] + c[ 1 ] * x[ 0 ] + c[ 2 ] * x[ 1 ] + c[ 3 ] * x[ 0 ] ** 2 + c[ 4 ] * x[ 0 ] * x[ 1 ] + c[ 5 ] * x[ 1 ] ** 2 )
        return float( self._img[ i ] )

    def contains( self, point ):
        return self.coeff_at( point ) != 0

    def min_position( self ):
        return self._min

    def max_position( self ):
        return self._max

    def display_boundaries_vtk( self, filename ):
        from .PowerDiagram import _write_vtk_polygons
        _write_vtk_polygons( filename, _piece_faces( "box", ( self._min, self._max ) ) )

    def pieces_half_spaces( self ):
        return [ _piece_half_spaces( "box", ( self._min, self._max ) ) + ( 1.0, ) ]

    def to_sdot( self, target_mass = 1.0 ):
        if self._cache is None:
            self._cache = {}
        if target_mass not in self._cache:
            import sdot
            # sdot's first axis is x: pysdot's last
            if self._poly is not None:
                self._cache[ target_mass ] = sdot.Image( self._bernstein(), degree = 2, frame = ( self._max - self._min ).tolist(),
                                                         origin = self._min.tolist(), target_mass = target_mass )
            else:
                self._cache[ target_mass ] = sdot.Image( np.ascontiguousarray( self._img.transpose() ), frame = self._max - self._min,
                                                         origin = self._min, target_mass = target_mass )
        return self._cache[ target_mass ]


# ---- helpers ---------------------------------------------------------------------------------------------------
def _boxes_as_image( pieces, d, tm ):
    """several boxes: an `Image` on the grid of all their coordinates, each pixel the sum of the coefficients of the boxes over it"""
    import sdot
    knots = [ np.unique( np.concatenate( [ np.array( [ mi[ a ], ma[ a ] ] ) for _, ( mi, ma ), _ in pieces ] ) ) for a in range( d ) ]
    values = np.zeros( [ len( k ) - 1 for k in knots ] )
    for _, ( mi, ma ), c in pieces:
        sl = tuple( slice( np.searchsorted( knots[ a ], mi[ a ] ), np.searchsorted( knots[ a ], ma[ a ] ) ) for a in range( d ) )
        values[ sl ] += c
    if values.min() < 0:
        raise ValueError( "ConvexPolyhedraAssembly: the boxes give a negative density somewhere" )
    steps = [ np.diff( k ) for k in knots ]
    if all( np.allclose( s, s[ 0 ], rtol = 1e-12, atol = 0 ) for s in steps ):
        lo = np.array( [ k[ 0 ] for k in knots ] )
        return sdot.Image( values, frame = np.array( [ k[ -1 ] - k[ 0 ] for k in knots ] ), origin = lo, target_mass = tm )
    # an irregular grid: the knots, padded with their last value ( a ragged axis )
    nk = max( len( k ) for k in knots )
    padded = np.stack( [ np.concatenate( [ k, np.full( nk - len( k ), k[ -1 ] ) ] ) for k in knots ] )
    return sdot.Image( values, knots = padded, pixel_frame = np.eye( d ), origin = np.zeros( d ), target_mass = tm )


def _box_simplices( mi, ma ):
    """the box cut into `d!` simplices ( Kuhn )"""
    import itertools
    d = len( mi )
    res = []
    for perm in itertools.permutations( range( d ) ):
        v = np.zeros( d )
        pts = [ v.copy() ]
        for a in perm:
            v[ a ] = 1
            pts.append( v.copy() )
        res.append( mi + np.array( pts ) * ( ma - mi ) )
    return res


def _polyhedron_vertices( directions, offsets ):
    """the vertices of `directions . x <= offsets` ( bounded ), by enumeration of the `d`-tuples of planes ( small polyhedra )"""
    import itertools
    d = directions.shape[ 1 ]
    res = []
    for idx in itertools.combinations( range( len( offsets ) ), d ):
        A = directions[ list( idx ) ]
        if abs( np.linalg.det( A ) ) < 1e-14:
            continue
        x = np.linalg.solve( A, offsets[ list( idx ) ] )
        if np.all( directions @ x <= offsets + 1e-10 * ( 1 + np.abs( offsets ) ) ):
            res.append( x )
    return np.unique( np.round( np.array( res ), 12 ), axis = 0 )


def _piece_polygon( kind, data ):
    """a 2D piece as a convex polygon, counter clockwise"""
    if kind == "box":
        mi, ma = data
        return np.array( [ [ mi[ 0 ], mi[ 1 ] ], [ ma[ 0 ], mi[ 1 ] ], [ ma[ 0 ], ma[ 1 ] ], [ mi[ 0 ], ma[ 1 ] ] ] )
    pts = data if kind == "simplex" else _polyhedron_vertices( *data )
    c = pts.mean( axis = 0 )
    return pts[ np.argsort( np.arctan2( pts[ :, 1 ] - c[ 1 ], pts[ :, 0 ] - c[ 0 ] ) ) ]


def _polygon_half_spaces( P ):
    """`( directions, offsets )` of a convex polygon, counter clockwise: its edges, outward"""
    e = np.roll( P, -1, axis = 0 ) - P
    n = np.stack( [ e[ :, 1 ], - e[ :, 0 ] ], axis = 1 )
    return n, ( n * P ).sum( axis = 1 )


def _difference( R, dirs, offs ):
    """the convex polygon `R` minus the convex set `dirs . x <= offs`, as disjoint convex polygons"""
    from .PowerDiagram import _clip_polygon
    res = []
    for k in range( len( offs ) ):
        piece = _clip_polygon( R, - dirs[ k : k + 1 ], - offs[ k : k + 1 ] )     # outside facet k...
        piece = _clip_polygon( piece, dirs[ :k ], offs[ :k ] )                     # ... inside the facets before
        if _area( piece ) > 0:
            res.append( piece )
    return res


def _area( P ):
    if len( P ) < 3:
        return 0.0
    return 0.5 * float( np.sum( P[ :, 0 ] * np.roll( P[ :, 1 ], -1 ) - np.roll( P[ :, 0 ], -1 ) * P[ :, 1 ] ) )


def _piece_half_spaces( kind, data ):
    """`( directions, offsets )` of a piece"""
    if kind == "box":
        mi, ma = data
        d = len( mi )
        return np.concatenate( [ np.eye( d ), - np.eye( d ) ] ), np.concatenate( [ ma, - mi ] )
    if kind == "simplex":
        from scipy.spatial import ConvexHull
        eq = ConvexHull( data ).equations                 # ( n . x + b <= 0 inside )
        return eq[ :, :-1 ], - eq[ :, -1 ]
    return data


def _piece_box( kind, data ):
    if kind == "box":
        return data
    pts = data if kind == "simplex" else _polyhedron_vertices( *data )
    return pts.min( axis = 0 ), pts.max( axis = 0 )


def _piece_volume( kind, data ):
    if kind == "box":
        return float( np.prod( data[ 1 ] - data[ 0 ] ) )
    if kind == "simplex":
        return abs( float( np.linalg.det( data[ 1: ] - data[ 0 ] ) ) ) / math.factorial( data.shape[ 1 ] )
    from scipy.spatial import ConvexHull
    return float( ConvexHull( _polyhedron_vertices( *data ) ).volume )


def _piece_contains( kind, data, x ):
    if kind == "box":
        return bool( np.all( x >= data[ 0 ] ) and np.all( x < data[ 1 ] ) )
    if kind == "simplex":
        bc = np.linalg.solve( ( data[ 1: ] - data[ 0 ] ).T, x - data[ 0 ] )
        return bool( np.all( bc >= 0 ) and bc.sum() <= 1 )
    return bool( np.all( data[ 0 ] @ x <= data[ 1 ] ) )


def _piece_faces( kind, data ):
    """the boundary of a piece as polygons ( lists of points ): its outline in 2D, its faces in 3D"""
    if kind == "box":
        mi, ma = data
        if len( mi ) == 2:
            return [ np.array( [ [ mi[ 0 ], mi[ 1 ] ], [ ma[ 0 ], mi[ 1 ] ], [ ma[ 0 ], ma[ 1 ] ], [ mi[ 0 ], ma[ 1 ] ] ] ) ]
        pts = _box_corners( mi, ma )
        return [ pts[ list( f ) ] for f in ( ( 0, 2, 3, 1 ), ( 4, 5, 7, 6 ), ( 0, 1, 5, 4 ), ( 2, 6, 7, 3 ), ( 0, 4, 6, 2 ), ( 1, 3, 7, 5 ) ) ]
    pts = data if kind == "simplex" else _polyhedron_vertices( *data )
    from scipy.spatial import ConvexHull
    h = ConvexHull( pts )
    if pts.shape[ 1 ] == 2:
        return [ pts[ h.vertices ] ]
    return [ pts[ s ] for s in h.simplices ]


def _box_corners( mi, ma ):
    return np.array( [ [ ( ma if ( i >> a ) & 1 else mi )[ a ] for a in range( 3 ) ] for i in range( 8 ) ] )
