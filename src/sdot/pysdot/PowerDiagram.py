"""`pysdot.PowerDiagram` on top of `sdot.PowerDiagram`.

The weights follow the same convention in both: the cell of seed `i` is where `|x - p_i|^2 - w_i` is smallest. What pysdot
returned is returned here, NOT normalized ( sdot normalizes the density to 1: its results are multiplied back by the mass
of the domain ), as numpy arrays, with the geometry cut in doubles.
"""
from .radial_funcs import RadialFuncUnit, sdot_radial_args
from .domain_types import ConvexPolyhedraAssembly
import numpy as np
import os


class DerResult:
    """what `der_integrals_wrt_weights` returns: a CSR matrix ( `m_offsets`, `m_columns`, `m_values` ), the integrals
    ( `v_values` ) and `error` ( 1 when a cell is empty and `stop_if_void` )"""
    def __init__( self, m_offsets, m_columns, m_values, v_values, error = 0 ):
        self.m_offsets = m_offsets
        self.m_columns = m_columns
        self.m_values = m_values
        self.v_values = v_values
        self.error = error


class PowerDiagram:
    def __init__( self, positions = None, weights = None, domain = None, radial_func = RadialFuncUnit(), replications = [] ):
        positions, weights, domain, radial_func = _old_call_forms( positions, weights, domain, radial_func )
        self.radial_func = radial_func

        self.positions = None
        self.weights = None
        self.domain = None
        self._replications = [ np.asarray( t, dtype = float ) for t in replications ]
        self._diagram = None

        if domain is not None:
            self.set_domain( domain )
        if positions is not None:
            self.set_positions( positions )
        if weights is not None:
            self.set_weights( weights )

    # ---- inputs ------------------------------------------------------------------------------------------------
    def add_replication( self, trans ):
        """for periodic boundaries: the seeds are repeated by the translation `trans` ( an sdot `periodicity` generator )"""
        self._replications.append( np.asarray( trans, dtype = float ) )
        self._diagram = None

    def get_positions( self ):
        return self.positions

    def set_positions( self, positions ):
        self.positions = np.ascontiguousarray( positions, dtype = float )
        self._diagram = None

    def get_weights( self ):
        return self.weights

    def set_weights( self, weights ):
        self.weights = np.ascontiguousarray( weights, dtype = float )
        self._diagram = None

    def get_domain( self ):
        return self.domain

    def set_domain( self, domain ):
        self.domain = domain
        self._diagram = None

    # ---- what is read ------------------------------------------------------------------------------------------
    def integrals( self ):
        """the integral of the density ( times the radial function ) over each cell, `[ n ]`"""
        return self._measures() * self._mass()

    def centroids( self, rand_ratio = 0.0 ):
        """`( 1 - rand_ratio ) * centroid + rand_ratio * random point` of each cell ( the seed for an empty cell )"""
        n, d = self.positions.shape
        mass = self._measures()
        first = np.stack( [ self._moment( k ) for k in range( d ) ], axis = 1 )
        res = self.positions.copy()
        ok = mass > 0
        res[ ok ] = first[ ok ] / mass[ ok, None ]
        if rand_ratio:
            res = ( 1 - rand_ratio ) * res + rand_ratio * self._random_points_in_cells()
        return res

    def second_order_moments( self ):
        """`int_{cell_i} |x - p_i|^2 rho`, `[ n ]`"""
        p = self.positions
        d = p.shape[ 1 ]
        first = np.stack( [ self._moment( k ) for k in range( d ) ], axis = 1 )
        second = self._moment( d )
        mass = self._measures()
        return ( second - 2 * ( p * first ).sum( axis = 1 ) + ( p * p ).sum( axis = 1 ) * mass ) * self._mass()

    def der_integrals_wrt_weights( self, stop_if_void = False ):
        """the derivatives of `integrals` with respect to the weights, as a CSR matrix ( sorted columns, the diagonal included )"""
        n = self.positions.shape[ 0 ]
        v_values = self.integrals()
        if stop_if_void and np.any( v_values == 0 ):
            return DerResult( np.zeros( n + 1, dtype = np.uint64 ), np.zeros( 0, dtype = np.uint64 ), np.zeros( 0 ), v_values, 1 )
        rows, cols, vals = self._jacobian_entries()
        from scipy.sparse import csr_matrix
        M = csr_matrix( ( vals * self._mass(), ( rows, cols ) ), shape = ( n, n ) )
        M.sum_duplicates()
        M.sort_indices()
        return DerResult( M.indptr.astype( np.uint64 ), M.indices.astype( np.uint64 ), M.data.astype( float ), v_values, 0 )

    def der_centroids_and_integrals_wrt_weight_and_positions( self ):
        """the derivatives of the centroids and of the integrals with respect to the positions and the weights, a CSR "system"
        ( `m_offsets`, `m_columns`, `m_values`, `v_values`, `error` ) with blocks `( d + 1 )^2`: rows `d centroid_x_i, ...,
        d integral_i`, columns `dirac_pos_x_j, ..., weight_j`. `v_values` holds the centroids and the integrals, interleaved
        the same way.

        Through the automatic differentiation of sdot: Jax or PyTorch must be installed. A few reverse sweeps per COLOR of the
        cells ( cells whose neighborhoods do not meet share a sweep ), a few tens in 2D."""
        from ._derivatives import der_centroids_and_integrals
        return der_centroids_and_integrals( self )

    def cell_polyhedra( self ):
        """2D: `( offsets, coordinates )`, the vertices of cell `i` being `coordinates[ offsets[ i ] : offsets[ i + 1 ] ]`, counter
        clockwise. 3D: `( offset_polyhedra, offset_polygons, coordinates )`."""
        cells = self._cells()
        d = self.positions.shape[ 1 ]
        if d == 2:
            offsets, coords = [ 0 ], []
            for vp in cells:
                coords += list( vp )
                offsets.append( len( coords ) )
            return np.array( offsets, dtype = np.uint64 ), np.array( coords, dtype = float ).reshape( -1, 2 )
        off_polyhedra, off_polygons, coords = [ 0 ], [ 0 ], []
        for faces in cells:
            for f in faces:
                coords += list( f )
                off_polygons.append( len( coords ) )
            off_polyhedra.append( len( off_polygons ) - 1 )
        return np.array( off_polyhedra, dtype = np.uint64 ), np.array( off_polygons, dtype = np.uint64 ), np.array( coords, dtype = float ).reshape( -1, 3 )

    def vtk_mesh_data( self, shrink_factor = 0.0 ):
        """`( cells, celltypes, points )` for `pyvista.UnstructuredGrid`: one polygon per cell in 2D, per face in 3D"""
        polys, _ = self._polygons( shrink_factor )
        cells, points = [], []
        for p in polys:
            cells += [ len( p ) ] + list( range( len( points ), len( points ) + len( p ) ) )
            points += list( _as_3d( p ) )
        return np.array( cells, dtype = np.int64 ), np.full( len( polys ), 7, dtype = np.uint8 ), np.array( points, dtype = float ).reshape( -1, 3 )

    def display_vtk( self, filename, points = False, centroids = False ):
        """the cells in a legacy VTK file ( one polygon per cell in 2D, per face in 3D, the field `num` the seed )"""
        polys, nums = self._polygons()
        extra = []
        if points:
            extra.append( self.positions )
        if centroids:
            extra.append( self.centroids() )
        _write_vtk_polygons( filename, polys, { "num": nums }, extra )

    def display_vtk_points( self, filename, points = None ):
        _write_vtk_polygons( filename, [], {}, [ self.positions if points is None else np.asarray( points, dtype = float ) ] )

    def display_jupyter( self, disp_centroids = True, disp_positions = True, disp_ids = True, disp_arrows = False, hide_after = -1 ):
        """the diagram in the notebook, with the sdot viewer ( `show` )"""
        return self.sdot_diagram().show()

    def display_asy( self, filename, preamble = "", closing = "", output_format = "pdf", linewidth = 0.02, dotwidth = 0.0, values = np.array( [] ),
                     colormap = "inferno", avoid_bounds = False, min_rf = 1, max_rf = 0 ):
        """the cells in an Asymptote file ( 2D ): each one filled with the color of `values[ i ]` ( in [ 0, 1 ], inferno ) when given,
        and drawn. `linewidth <= 0` and `dotwidth`: the seeds as dots instead. `avoid_bounds` and `min_rf` / `max_rf` ( the shading
        of a radial function ) are ignored."""
        if self.positions.shape[ 1 ] != 2:
            raise NotImplementedError( "display_asy: 2D only" )
        dn = os.path.dirname( filename )
        if dn:
            os.makedirs( dn, exist_ok = True )
        values = np.asarray( values, dtype = float ).reshape( -1 )

        def rgb( v ):
            import matplotlib
            c = matplotlib.colormaps[ colormap ]( float( np.clip( v, 0, 1 ) ) )
            return "rgb({},{},{})".format( c[ 0 ], c[ 1 ], c[ 2 ] )

        def path( p ):
            return "--".join( "({:.16g},{:.16g})".format( x[ 0 ], x[ 1 ] ) for x in p ) + "--cycle"

        p = "settings.outformat = \"{}\";\nunitsize(1cm);\n".format( output_format )
        if linewidth > 0:
            p += "defaultpen({}cm);\n".format( linewidth )
        elif dotwidth > 0:
            p += "defaultpen({}cm);\n".format( dotwidth / 6 )
        out = [ p + preamble ]
        if linewidth <= 0 and dotwidth:
            for i, x in enumerate( self.positions ):
                out.append( "dot(({},{}){});\n".format( x[ 0 ], x[ 1 ], "," + rgb( values[ i ] ) if values.size else "" ) )
        else:
            polys, nums = self._polygons()
            for poly, i in zip( polys, nums ):
                if values.size:
                    out.append( "fill({},{});\n".format( path( poly ), rgb( values[ i ] ) ) )
                out.append( "draw({});\n".format( path( poly ) ) )
        out.append( closing )
        with open( filename, "w" ) as f:
            f.write( "".join( out ) )

    def image_integrals( self, min_pos, max_pos, dimensions ):
        raise NotImplementedError( "sdot.pysdot: image_integrals is not available -- `sdot_diagram().cells` gives the geometry" )

    def distances_from_boundaries( self, points, count_domain_boundaries = False ):
        """for each point, its distance to the nearest edge of a cell ( 2D ), NEGATIVE for a point inside the domain ( as in pysdot ) --
        the edges on the boundary of the domain only with `count_domain_boundaries`"""
        if self.positions.shape[ 1 ] != 2:
            raise NotImplementedError( "distances_from_boundaries: 2D only" )
        pts = np.asarray( points, dtype = float ).reshape( -1, 2 )
        polys, _ = self._polygons()
        a = np.concatenate( [ p for p in polys ] )
        b = np.concatenate( [ np.roll( p, -1, axis = 0 ) for p in polys ] )
        if not count_domain_boundaries:
            # an edge on a line of the boundary of a piece of the domain
            on_bnd = np.zeros( len( a ), bool )
            scale = float( np.max( np.ptp( a, axis = 0 ) ) ) if len( a ) else 1.0
            for dirs, offs, _ in ( self.domain.pieces_half_spaces() if hasattr( self.domain, "pieces_half_spaces" ) else [] ):
                for n, o in zip( dirs, offs ):
                    nn = np.linalg.norm( n )
                    on_bnd |= ( np.abs( a @ n - o ) <= 1e-10 * scale * nn ) & ( np.abs( b @ n - o ) <= 1e-10 * scale * nn )
            a, b = a[ ~on_bnd ], b[ ~on_bnd ]
        res = np.full( len( pts ), np.finfo( float ).max )
        if not len( a ):
            return res
        ab = b - a
        l2 = np.maximum( ( ab * ab ).sum( axis = 1 ), 1e-300 )
        for c in range( 0, len( pts ), 1024 ):
            p = pts[ c : c + 1024, None, : ]
            t = np.clip( ( ( p - a ) * ab ).sum( axis = 2 ) / l2, 0, 1 )
            dist = np.linalg.norm( p - ( a + t[ :, :, None ] * ab ), axis = 2 )
            res[ c : c + 1024 ] = dist.min( axis = 1 )
        inside = np.array( [ bool( self.domain.contains( p ) ) for p in pts ] )
        return np.where( inside, - res, res )

    # ---- sdot ------------------------------------------------------------------------------------------------
    def sdot_diagram( self ):
        """the `sdot.PowerDiagram` of the current positions, weights, domain and radial function"""
        if self._diagram is None:
            import sdot
            if self.positions is None:
                raise RuntimeError( "positions in PowerDiagram must be specified" )
            if self.domain is None:
                self.set_domain( default_domain( self.positions.shape[ 1 ] ) )
            if self.weights is None:
                self.set_weights( np.ones( self.positions.shape[ 0 ] ) )
            sink_weight, radial = sdot_radial_args( self.radial_func )
            kw = dict( distribution = self.domain.to_sdot(), kernel_dtype = "FP64", sink_weight = sink_weight, radial_func = radial )
            self._images = None
            try:
                self._diagram = sdot.PowerDiagram( self.positions, self.weights, periodicity = self.sdot_periodicity(), **kw )
            except ValueError as e:
                if "fundamental domain" not in str( e ):
                    raise
                # the domain is not a fundamental domain of the translations ( it overlaps its images ): what pysdot did, literally --
                # the translated seeds are more seeds, and only the cells of the given ones are reported
                self._images = self.sdot_periodicity()
                pos, w = self.extended_seeds( self.positions, self.weights, np.concatenate )
                self._diagram = sdot.PowerDiagram( pos, w, **kw )
        return self._diagram

    def extended_seeds( self, positions, weights, concatenate ):
        """the seeds and their images by the replications, when they are explicit ( see `sdot_diagram` )"""
        if not self._images:
            return positions, weights
        return ( concatenate( [ positions ] + [ positions + np.asarray( t, dtype = float ) for t in self._images ] ),
                 concatenate( [ weights ] * ( 1 + len( self._images ) ) ) )

    def sdot_periodicity( self ):
        """the replications as an sdot `periodicity` ( the generators of a group of translations ), or `None`"""
        gens = [ t for t in self._replications if np.any( t != 0 ) ]
        return gens or None

    def _mass( self ):
        if self.domain is None:
            self.set_domain( default_domain( self.positions.shape[ 1 ] ) )
        return float( self.domain.measure() )

    def _memo( self, key, compute ):
        """`compute( diagram )`, once per diagram: what is read again without a change of the inputs ( `set_*` ) is not recomputed"""
        pd = self.sdot_diagram()
        if getattr( self, "_memo_of", None ) is not pd:
            self._memo_of, self._memo_values = pd, {}
        if key not in self._memo_values:
            self._memo_values[ key ] = compute( pd )
        return self._memo_values[ key ]

    def _measures( self ):
        n = self.positions.shape[ 0 ]
        return self._memo( "measures", lambda pd: np.asarray( pd.measures, dtype = float ).reshape( -1 )[ : n ] )

    def _moment( self, k ):
        n = self.positions.shape[ 0 ]
        return self._memo( ( "moment", k ), lambda pd: np.asarray( pd.moment( k ), dtype = float ).reshape( -1 )[ : n ] )

    def _jacobian_entries( self ):
        """`( rows, cols, vals )` of `d measures / d weights` ( normalized, duplicates not summed ): `-v` off the diagonal for each
        facet, `+v` on it -- and the cell's own slot ( a radial function, the sphere of a sink ) on the diagonal only"""
        n = self.positions.shape[ 0 ]
        nb, ids, vals = self._memo( "hessian_rows", lambda pd: pd.hessian_rows() )
        nb, ids, vals = np.asarray( nb ).reshape( -1 ), np.asarray( ids ), np.asarray( vals, dtype = float )
        N = nb.size                                       # ( more than `n` with explicit images: their rows are not reported )
        cap = ids.shape[ 1 ] if ids.ndim == 2 else 0
        i = np.repeat( np.arange( N ), cap ).reshape( N, cap )
        live = ( np.arange( cap )[ None, : ] < nb[ :, None ] ) & ( ids >= 0 ) & ( i < n )
        i, j, v = i[ live ], ids[ live ], vals[ live ]
        own = j == i                                      # the cell's own slot, not a facet
        i, j, v, own = i[ own | ( v != 0 ) ], j[ own | ( v != 0 ) ], v[ own | ( v != 0 ) ], own[ own | ( v != 0 ) ]   # ( a facet out of the balls of a sink: not in the matrix, as in pysdot )
        j = j % n                                         # ( an image of a seed by a periodicity )
        rows = np.concatenate( [ i[ ~own ], i ] )
        cols = np.concatenate( [ j[ ~own ], i ] )
        vs = np.concatenate( [ - v[ ~own ], v ] )
        return rows, cols, vs

    def _cells( self ):
        """per cell: its vertices counter clockwise in 2D, its faces ( lists of points, outward ) in 3D"""
        cells = self._memo( "cells", lambda pd: pd.cells )
        d = self.positions.shape[ 1 ]
        res = []
        all_faces = cells.faces
        if not isinstance( all_faces, list ) or ( len( all_faces ) and not isinstance( all_faces[ 0 ], list ) ):
            all_faces = [ all_faces ]
        for i in range( self.positions.shape[ 0 ] ):
            vp = np.asarray( cells.vertices( i ), dtype = float )
            if d == 2:
                if len( vp ):
                    c = vp.mean( axis = 0 )
                    vp = vp[ np.argsort( np.arctan2( vp[ :, 1 ] - c[ 1 ], vp[ :, 0 ] - c[ 0 ] ) ) ]
                res.append( vp )
            else:
                c = vp.mean( axis = 0 ) if len( vp ) else np.zeros( 3 )
                faces = []
                for f in all_faces[ i ]:
                    pts = vp[ list( f ) ]
                    if len( pts ) < 3:
                        continue
                    fc = pts.mean( axis = 0 )
                    nrm = fc - c
                    nrm /= np.linalg.norm( nrm ) or 1
                    e0 = pts[ 0 ] - fc
                    e0 -= nrm * ( e0 @ nrm )
                    e0 /= np.linalg.norm( e0 ) or 1
                    e1 = np.cross( nrm, e0 )
                    faces.append( pts[ np.argsort( np.arctan2( ( pts - fc ) @ e1, ( pts - fc ) @ e0 ) ) ] )
                res.append( faces )
        return res

    def _polygons( self, shrink_factor = 0.0 ):
        """`( polygons, seed of each )`: the cells in 2D, their faces in 3D, shrunk toward the centre of their cell"""
        polys, nums = [], []
        d = self.positions.shape[ 1 ]
        if d == 2:
            # as in pysdot: each cell cut by each piece of the domain ( a domain that is not convex ), and by its ball ( a sink )
            pieces = self.domain.pieces_half_spaces() if hasattr( self.domain, "pieces_half_spaces" ) else [ None ]
            balls = self.radial_func.ball_cut()
        for i, c in enumerate( self._cells() ):
            if d == 2:
                fs = []
                for piece in pieces:
                    f = c if piece is None else _clip_polygon( c, piece[ 0 ], piece[ 1 ] )
                    if balls and len( f ):
                        f = _clip_polygon( f, *_ball_half_spaces( self.positions[ i ], self.weights[ i ] ) )
                    if len( f ) >= 3:
                        fs.append( f )
            else:
                fs = c
            if not len( fs ) or not len( fs[ 0 ] ):
                continue
            ctr = np.concatenate( fs ).mean( axis = 0 )
            for f in fs:
                polys.append( ctr + ( 1 - shrink_factor ) * ( f - ctr ) )
                nums.append( i )
        return polys, nums

    def _random_points_in_cells( self ):
        """a point drawn uniformly in each cell ( a simplex of the cell by its volume, then a point of it; the seed for an empty cell )"""
        import math
        d = self.positions.shape[ 1 ]

        def simplices( pd ):
            # per cell: its simplices `[ k, d + 1, d ]` and their cumulated volumes ( the cell's own geometry: a triangulation of a convex set )
            from scipy.spatial import Delaunay
            cells = self._memo( "cells", lambda pd: pd.cells )
            res = []
            for i in range( self.positions.shape[ 0 ] ):
                vp = np.asarray( cells.vertices( i ), dtype = float )
                if len( vp ) <= d:
                    res.append( None )
                    continue
                s = vp[ Delaunay( vp ).simplices ]
                vol = np.abs( np.linalg.det( s[ :, 1: ] - s[ :, :1 ] ) ) / math.factorial( d )
                res.append( ( s, np.cumsum( vol ) / vol.sum() ) )
            return res

        rng = np.random.default_rng()
        res = self.positions.copy()
        for i, sv in enumerate( self._memo( "simplices", simplices ) ):
            if sv is not None:
                s, cum = sv
                k = min( int( np.searchsorted( cum, rng.uniform() ) ), len( s ) - 1 )
                res[ i ] = rng.dirichlet( np.ones( d + 1 ) ) @ s[ k ]
        return res


def _old_call_forms( positions, weights, domain, radial_func ):
    """the call forms of older pysdots, which some programs still use: `( domain, radial_func )` or `( domain )` in the place of
    `( positions, weights )`"""
    if hasattr( positions, "to_sdot" ):
        if domain is None:
            domain = positions
        positions = None
        if weights is not None and hasattr( weights, "ball_cut" ):
            radial_func = weights
            weights = None
    return positions, weights, domain, radial_func


def default_domain( d ):
    """the unit box, the domain of pysdot when there is none"""
    domain = ConvexPolyhedraAssembly()
    domain.add_box( np.zeros( d ), np.ones( d ) )
    return domain


def _clip_polygon( poly, directions, offsets ):
    """the convex polygon `poly` ( counter clockwise ) cut by the half-planes `directions . x <= offsets`"""
    poly = np.asarray( poly, dtype = float )
    directions, offsets = np.asarray( directions, dtype = float ), np.asarray( offsets, dtype = float )
    cut = np.any( poly @ directions.T > offsets, axis = 0 )       # ( the half-planes that remove something: often none )
    for n, o in zip( directions[ cut ], offsets[ cut ] ):
        if not len( poly ):
            break
        s = poly @ n - o
        res = []
        for k in range( len( poly ) ):
            a, b, sa, sb = poly[ k ], poly[ ( k + 1 ) % len( poly ) ], s[ k ], s[ ( k + 1 ) % len( poly ) ]
            if sa <= 0:
                res.append( a )
            if ( sa < 0 < sb ) or ( sb < 0 < sa ):
                res.append( a + ( b - a ) * sa / ( sa - sb ) )
        poly = np.array( res ).reshape( -1, 2 )
    return poly


def _ball_half_spaces( center, weight, nb = 64 ):
    """the ball `|x - center|^2 <= weight`, as the `nb` half-planes of a polygon around it"""
    r = np.sqrt( max( float( weight ), 0.0 ) ) / np.cos( np.pi / nb )
    a = 2 * np.pi * np.arange( nb ) / nb
    dirs = np.stack( [ np.cos( a ), np.sin( a ) ], axis = 1 )
    return dirs, dirs @ np.asarray( center, dtype = float ) + r * np.cos( np.pi / nb )


def _as_3d( p ):
    p = np.asarray( p, dtype = float ).reshape( len( p ), -1 )
    return np.concatenate( [ p, np.zeros( ( len( p ), 3 - p.shape[ 1 ] ) ) ], axis = 1 ) if p.shape[ 1 ] < 3 else p


def _write_vtk_polygons( filename, polygons, cell_fields = {}, point_sets = [] ):
    """a legacy ASCII VTK file: `polygons` ( lists of points ), plus one vertex per point of `point_sets`"""
    dn = os.path.dirname( filename )
    if dn:
        os.makedirs( dn, exist_ok = True )
    points = [ _as_3d( p ) for p in polygons ] + [ _as_3d( p ) for p in point_sets ]
    pts = np.concatenate( points ) if points else np.zeros( ( 0, 3 ) )
    cells, o = [], 0
    for p in polygons:
        cells.append( [ len( p ) ] + list( range( o, o + len( p ) ) ) )
        o += len( p )
    nb_poly = len( cells )
    for p in point_sets:
        for _ in range( len( p ) ):
            cells.append( [ 1, o ] )
            o += 1
    with open( filename, "w" ) as f:
        f.write( "# vtk DataFile Version 3.0\nsdot.pysdot\nASCII\nDATASET UNSTRUCTURED_GRID\n" )
        f.write( f"POINTS { len( pts ) } float\n" )
        for x in pts:
            f.write( f"{ x[ 0 ] } { x[ 1 ] } { x[ 2 ] }\n" )
        f.write( f"CELLS { len( cells ) } { sum( len( c ) for c in cells ) }\n" )
        for c in cells:
            f.write( " ".join( map( str, c ) ) + "\n" )
        f.write( f"CELL_TYPES { len( cells ) }\n" )
        for k in range( len( cells ) ):
            f.write( "7\n" if k < nb_poly else "1\n" )
        if cell_fields and len( cells ):
            f.write( f"CELL_DATA { len( cells ) }\n" )
            for name, values in cell_fields.items():
                v = list( values ) + [ -1 ] * ( len( cells ) - nb_poly )
                f.write( f"SCALARS { name } float 1\nLOOKUP_TABLE default\n" )
                for x in v:
                    f.write( f"{ x }\n" )
