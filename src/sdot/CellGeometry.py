"""THE NUMBERS OF A CELL COMPLEX, as functions of its seeds: tensors of the framework of the positions ( numpy, jax, torch... ),
differentiable with respect to the positions and the weights ( `sdot/complex/Geometry.h`, forward and adjoint ).

    geo = cx.geometry()                              # the positions and weights of the diagram
    geo = cx.geometry( positions, weights )          # others, with the same topology -- under a `jax.jit`, in a descent
    geo.measures( k ), geo.centroids( k ), geo.normals, geo.seed_vectors, geo.vertex_positions, geo.simplices( k )
    geo.boundary_distances( points, cells, facets )

The TOPOLOGY is a host constant ( `CellComplex`, built from values ): what changes with the seeds here is the geometry of that topology,
not the topology. A vertex is computed from its generators ( the bisectors and walls it is the corner of ), once per copy, and a face
from the simplices of its fan: both sides of a facet read the same numbers by construction.

= Internals

The tables ( `CellComplex._tables` ): the ROWS ( one per local vertex that is a corner: its seed, its `d` cuts, the frame of its cell ),
the APEXES ( per local face of dimension 2 or more, the rows of its vertices ), and per dimension `k` the SIMPLICES of the fans of the
representatives of the faces ( `k + 1` corners: a row, or `-1 - apex` ), with the sign that orients the area vector of a facet from its
first side to its second.
"""

import numpy as np

import loom
from loom.compilation.FfiCode import FfiCode
from loom.tensor.host import to_host
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, new_batch_axis
from loom.util import Aggregate


_INCLUDES = [ "sdot/complex/Geometry.h" ]


class _Rows( Aggregate ):
    eq_a     : IntTensor [ "num_cut", dict( size = 32 ) ]
    eq_ea    : IntTensor [ "num_cut", dict( size = 32 ) ]
    eq_b     : IntTensor [ "num_cut", dict( size = 32 ) ]
    eq_eb    : IntTensor [ "num_cut", dict( size = 32 ) ]
    frame    : RealTensor[ "dim", "aff" ]
    num_cut  : Axis[ "nb_cuts" ]
    dim      : Axis[ "nb_dims" ]
    aff      : Axis[ "nb_affs" ]
    nb_cuts  : CtShapeVar
    nb_dims  : CtShapeVar
    nb_affs  : CtShapeVar


class _Point( Aggregate ):
    x        : RealTensor[ "dim" ]
    dim      : Axis[ "nb_dims" ]
    nb_dims  : CtShapeVar


class _Face( Aggregate ):
    s0       : IntTensor [ dict( size = 32 ) ]
    s1       : IntTensor [ dict( size = 32 ) ]


class _FaceOut( Aggregate ):
    measure  : RealTensor
    centroid : RealTensor[ "dim" ]
    area     : RealTensor[ "dim" ]
    dim      : Axis[ "nb_dims" ]
    nb_dims  : CtShapeVar


class _SimplexOut( Aggregate ):
    points   : RealTensor[ "corner", "dim" ]
    corner   : Axis[ "nb_corners" ]
    dim      : Axis[ "nb_dims" ]
    nb_corners : CtShapeVar
    nb_dims  : CtShapeVar


class _DistIn( Aggregate ):
    x        : RealTensor[ "dim" ]
    facet    : IntTensor [ dict( size = 32 ) ]
    sign     : RealTensor
    dim      : Axis[ "nb_dims" ]
    nb_dims  : CtShapeVar


class _DistOut( Aggregate ):
    distance : RealTensor


class _SeedVec( Aggregate ):
    a        : IntTensor [ dict( size = 32 ) ]
    j        : IntTensor [ dict( size = 32 ) ]
    t        : IntTensor [ dict( size = 32 ) ]
    sign     : RealTensor
    frame    : RealTensor[ "dim", "dir" ]
    dim      : Axis[ "nb_dims" ]
    dir      : Axis[ "nb_dims" ]
    nb_dims  : CtShapeVar


class _VecOut( Aggregate ):
    vector   : RealTensor[ "dim" ]
    dim      : Axis[ "nb_dims" ]
    nb_dims  : CtShapeVar


def _ax( n, name ):
    return Axis( ShapeVar( int( n ) ), name = name )


def _first( raw, n ):
    """the first `n` rows of a batched output ( a device rounds its batches up )"""
    return raw[ :n ]


class CellGeometry:
    """The numbers of a `CellComplex` for given seeds ( `CellComplex.geometry` ): tensors of the framework of the positions,
    differentiable with respect to the positions and the weights. The faces are those of the complex; their geometry is that of
    these seeds."""

    def __init__( self, complex, positions, weights = None ):
        self.complex = complex
        cx = complex
        d = cx.dim
        self._d = d
        raw = getattr( positions, "raw", positions )
        self._ops = loom.ops( raw if raw is not None else np.zeros( 1 ) )
        self.positions = raw
        self.weights = getattr( weights, "raw", weights )
        # the frame of the cuts: the folded seeds with a periodicity ( the folding of the snapshot, a constant map per seed )
        self._P = raw if cx._fold is None or raw is None else _affine( raw, cx._fold )
        self._cache = {}

    # ---- the kernels --------------------------------------------------------------------------------

    def _vertices( self ):
        """`[ nb_rows, d ]`: the position of each row ( a local vertex that is a corner ), in the frame of its cell's copy"""
        if "X" in self._cache:
            return self._cache[ "X" ]
        T, d = self.complex._tables(), self._d
        if self.positions is None:                       # ( cells without seeds: their vertices as they were cut, constants )
            self._cache[ "X" ] = self.complex._V[ T[ "rows" ] ]
            return self._cache[ "X" ]
        m = len( T[ "own" ] )
        num_row = new_batch_axis( m, prefix = "row" )
        rows = _Rows( nb_cuts = d, nb_dims = d, nb_affs = d + 1, batch_axes = [ num_row ] )
        for key in ( "eq_a", "eq_ea", "eq_b", "eq_eb" ):
            getattr( rows, key ).set( np.ascontiguousarray( T[ key ], dtype = np.int32 ) )
        rows.frame.set( T[ "frame" ] )
        out = _Point( nb_dims = d, batch_axes = [ num_row ] )
        n = self.complex._n
        num_seed, dim = _ax( n, "num_seed" ), _ax( d, "dim" )
        maps, walls = T[ "maps" ], T[ "walls" ]
        kw = dict( positions = RealTensor[ num_seed, dim ]( self._P ), rows = rows, out = loom.out( out ),
                   maps = RealTensor[ _ax( len( maps ), "num_map" ), dim, _ax( d + 1, "aff" ) ]( maps ),
                   walls = RealTensor[ _ax( len( walls ), "num_wall" ), _ax( d + 1, "aff" ) ]( walls ) )
        if self.weights is not None:
            kw[ "weights" ] = RealTensor[ num_seed ]( self.weights )
            wexpr = "inputs.weights"
        else:
            wexpr = "sdot::NoneTensor<double,sdot::Tuple<>,sdot::Tuple<>>()"
        loom.ffi_call(
            "cell_complex_vertices",
            FfiCode.per_item( code = f"sdot::cell_complex::vertex_fwd<{ d }>( inputs.rows( batch_index ), inputs.positions, { wexpr }, "
                                     "inputs.maps, inputs.walls, outputs.out( batch_index ) );", includes = _INCLUDES ),
            FfiCode.per_item( f"sdot::cell_complex::vertex_bwd<{ d }>( inputs.rows( batch_index ), inputs.positions, { wexpr }, inputs.maps, "
                              "inputs.walls, grad_of_outputs.out( batch_index ), grad_of_inputs.positions, "
                              + ( "grad_of_inputs.weights" if self.weights is not None else wexpr ) + " );",
                              computed_grads = "computed_grads.from_out_x.positions = true;"
                                               + ( " computed_grads.from_out_x.weights = true;" if self.weights is not None else "" ),
                              includes = _INCLUDES ),
            **kw )
        X = _first( out.x.raw, m )
        self._cache[ "X" ] = X
        return X

    def _faces( self, k ):
        """`( measures [ m ], centroids [ m, d ], areas [ m, d ] )` of the bounded `k`-faces ( `k >= 1` ), from their fans"""
        if ( "faces", k ) in self._cache:
            return self._cache[ ( "faces", k ) ]
        T, d = self.complex._tables(), self._d
        S = T[ "simplices" ][ k ]
        X = self._vertices()
        nf = len( S[ "s0" ] )
        num_face = new_batch_axis( max( nf, 1 ), prefix = "face" )
        faces = _Face( batch_axes = [ num_face ] )
        faces.s0.set( S[ "s0" ] if nf else np.zeros( 1, np.int32 ) )
        faces.s1.set( S[ "s1" ] if nf else np.zeros( 1, np.int32 ) )
        out = _FaceOut( nb_dims = d, batch_axes = [ num_face ] )
        corners, signs = S[ "corners" ], S[ "signs" ]
        ns = max( len( corners ), 1 )
        if len( corners ) == 0:
            corners, signs = np.zeros( ( 1, k + 1 ), np.int32 ), np.ones( 1 )
        args = self._apex_args( X )
        loom.ffi_call(
            f"cell_complex_faces_{ k }",
            FfiCode.per_item( code = f"sdot::cell_complex::face_fwd<{ k },{ d }>( inputs.faces( batch_index ), inputs.X, inputs.corners, "
                                     "inputs.signs, inputs.apex_off, inputs.apex_rows, outputs.out( batch_index ) );", includes = _INCLUDES ),
            FfiCode.per_item( f"sdot::cell_complex::face_bwd<{ k },{ d }>( inputs.faces( batch_index ), inputs.X, inputs.corners, inputs.signs, "
                              "inputs.apex_off, inputs.apex_rows, outputs.out( batch_index ), grad_of_outputs.out( batch_index ), "
                              "grad_of_inputs.X );",
                              computed_grads = "computed_grads.from_out_measure.X = true; computed_grads.from_out_centroid.X = true; "
                                               "computed_grads.from_out_area.X = true;", includes = _INCLUDES ),
            faces = faces, out = loom.out( out ),
            corners = IntTensor[ _ax( ns, "num_simplex" ), _ax( k + 1, "num_corner" ), dict( size = 32 ) ]( corners.astype( np.int32 ) ),
            signs = RealTensor[ _ax( ns, "num_simplex" ) ]( signs.astype( float ) ),
            **args )
        res = ( _first( out.measure.raw, nf ), _first( out.centroid.raw, nf ), _first( out.area.raw, nf ) )
        self._cache[ ( "faces", k ) ] = res
        return res

    def _apex_args( self, X ):
        T, d = self.complex._tables(), self._d
        ao, ar = T[ "apex_off" ], T[ "apex_rows" ]
        if len( ar ) == 0:
            ar = np.zeros( 1, np.int32 )
        return dict( X = RealTensor[ _ax( len( T[ "own" ] ), "num_row" ), _ax( d, "dim" ) ]( X ),
                     apex_off = IntTensor[ _ax( len( ao ), "num_apex_off" ), dict( size = 32 ) ]( ao.astype( np.int32 ) ),
                     apex_rows = IntTensor[ _ax( len( ar ), "num_apex_row" ), dict( size = 32 ) ]( ar.astype( np.int32 ) ) )

    def _points( self, corners ):
        """`[ m, K + 1, d ]`: the corners of simplices ( rows or apexes ) as points"""
        d = self._d
        m, K1 = corners.shape
        X = self._vertices()
        num = new_batch_axis( max( m, 1 ), prefix = "simplex" )
        out = _SimplexOut( nb_corners = K1, nb_dims = d, batch_axes = [ num ] )
        if m == 0:
            corners = np.zeros( ( 1, K1 ), np.int32 )
        loom.ffi_call(
            f"cell_complex_points_{ K1 - 1 }",
            FfiCode.per_item( code = f"sdot::cell_complex::simplex_fwd<{ K1 - 1 },{ d }>( SI( flat_index ), inputs.X, inputs.corners, "
                                     "inputs.apex_off, inputs.apex_rows, outputs.out( batch_index ) );", includes = _INCLUDES ),
            FfiCode.per_item( f"sdot::cell_complex::simplex_bwd<{ K1 - 1 },{ d }>( SI( flat_index ), inputs.corners, inputs.apex_off, "
                              "inputs.apex_rows, grad_of_outputs.out( batch_index ), grad_of_inputs.X );",
                              computed_grads = "computed_grads.from_out_points.X = true;", includes = _INCLUDES ),
            out = loom.out( out ),
            corners = IntTensor[ _ax( max( m, 1 ), "num_simplex" ), _ax( K1, "num_corner" ), dict( size = 32 ) ]( corners.astype( np.int32 ) ),
            **self._apex_args( X ) )
        return _first( out.points.raw, m )

    # ---- what a user reads ----------------------------------------------------------------------------

    @property
    def vertex_positions( self ):
        """`[ nb_vertices, d ]`: the position of each vertex ( with a periodicity, of one of its copies )."""
        T = self.complex._tables()
        return self._points( T[ "vertex_rows" ][ :, None ] )[ :, 0 ]

    def measures( self, k ):
        """`[ nb_faces( k ) ]`: length, area, volume... ( `1` for a vertex, infinite for an unbounded face )."""
        return self._full( k )[ 0 ]

    def centroids( self, k ):
        """`[ nb_faces( k ), d ]`: the centers of mass ( NaN for an unbounded face )."""
        return self._full( k )[ 1 ]

    @property
    def normals( self ):
        """`[ nb_facets, d ]`: the unit normals of the facets, from their first side to their second."""
        d = self._d
        if d < 2:
            return self._ops.array( self.complex._sides_and_normals()[ 1 ], self._ops.dtype_of( self.positions ) )
        # the bounded facets by their area vectors, the others ( unbounded ) by the direction to the seed across -- exact for a bisector
        S = self.complex._tables()[ "simplices" ][ d - 1 ]
        faces = S[ "faces" ]
        m = self.complex.nb_faces( d - 1 )
        a = self._faces( d - 1 )[ 2 ]
        na = a / self._ops.sqrt( self._ops.sum( a * a, axis = 1 ) )[ :, None ]
        if len( faces ) == m:
            return na
        v = self.seed_vectors
        nv = v / self._ops.sqrt( self._ops.sum( v * v, axis = 1 ) + _host_like( v[ :, 0 ], ( np.linalg.norm( np.asarray( to_host( v ) ), axis = 1 ) == 0 ).astype( float ) ) )[ :, None ]
        pos = np.arange( m ) + len( faces )
        pos[ faces ] = np.arange( len( faces ) )
        return self._ops.concatenate( [ na, nv ], axis = 0 )[ pos ] if not isinstance( na, np.ndarray ) else np.concatenate( [ na, nv ] )[ pos ]

    @property
    def seed_vectors( self ):
        """`[ nb_facets, d ]`: from the seed `sides[ 0 ]` to ( the image of ) the seed `sides[ 1 ]` across each facet, zero on a wall."""
        cx, d = self.complex, self._d
        T = cx._tables()
        F = T[ "seed_vectors" ]
        m = len( F[ "a" ] )
        num = new_batch_axis( max( m, 1 ), prefix = "facet" )
        inp = _SeedVec( nb_dims = d, batch_axes = [ num ] )
        pad = lambda x, v: x if m else np.full( ( 1, ) + np.shape( x )[ 1: ], v )
        inp.a.set( pad( F[ "a" ], 0 ) ); inp.j.set( pad( F[ "j" ], 0 ) ); inp.t.set( pad( F[ "t" ], 0 ) )
        inp.sign.set( pad( F[ "sign" ], 1.0 ) ); inp.frame.set( pad( F[ "frame" ], 0.0 ) )
        out = _VecOut( nb_dims = d, batch_axes = [ num ] )
        n = cx._n
        maps = T[ "maps" ]
        loom.ffi_call(
            "cell_complex_seed_vectors",
            FfiCode.per_item( code = f"sdot::cell_complex::seed_vector_fwd<{ d }>( inputs.fc( batch_index ), inputs.positions, inputs.maps, "
                                     "outputs.out( batch_index ) );", includes = _INCLUDES ),
            FfiCode.per_item( f"sdot::cell_complex::seed_vector_bwd<{ d }>( inputs.fc( batch_index ), inputs.maps, grad_of_outputs.out( batch_index ), "
                              "grad_of_inputs.positions );",
                              computed_grads = "computed_grads.from_out_vector.positions = true;", includes = _INCLUDES ),
            fc = inp, out = loom.out( out ),
            positions = RealTensor[ _ax( n, "num_seed" ), _ax( d, "dim" ) ]( self._P ),
            maps = RealTensor[ _ax( len( maps ), "num_map" ), _ax( d, "dim" ), _ax( d + 1, "aff" ) ]( maps ) )
        v = _first( out.vector.raw, m )
        wall = F[ "wall" ]
        if wall.any():                                   # ( zero, not NaN: a NaN would poison the derivatives of what uses the others )
            v = v * _host_like( v, ( ~wall ).astype( float )[ :, None ] )
        return v

    def simplices( self, k = None ):
        """`( points [ m, k + 1, d ], faces [ m ] )`: the faces of dimension `k` ( the cells by default ) cut into simplices ( see
        `CellComplex.simplices` ), the points a tensor of the framework of the positions."""
        cx = self.complex
        k = cx.dim if k is None else cx._check_dim( k )
        if k == 0:
            return self.vertex_positions[ :, None, : ], np.arange( cx.nb_faces( 0 ) )
        S = cx._tables()[ "simplices" ][ k ]
        keep = S[ "keep" ]
        return self._points( S[ "corners" ][ keep ] ), S[ "owner" ][ keep ]

    def boundary_distances( self, points, cells = None ):
        """`( distances [ m ], facets [ m ] )`: the distance from each point to the boundary of its cell ( positive inside ) and the
        nearest facet, differentiable with respect to the points and the seeds ( the facet chosen on the values, a constant of the
        derivative ). `cells` / `points` as in `CellComplex.boundary_distances`."""
        cx, d = self.complex, self._d
        raw = getattr( points, "raw", points )
        host = np.asarray( to_host( raw ), dtype = float ).reshape( -1, d )
        if cells is None:
            cells, _, maps = cx._locate( host )
            moved = _affine( raw, maps )
        else:
            moved = raw
        cells = np.asarray( cells, dtype = np.int64 ).reshape( -1 )
        hd, facets, lfac = cx._nearest_local_facets( np.asarray( to_host( moved ), dtype = float ).reshape( -1, d ), cells )
        ok = facets >= 0
        m = len( host )
        area, cen = self._local_facets( lfac[ ok ] )
        num = new_batch_axis( max( int( ok.sum() ), 1 ), prefix = "point" )
        pts = _DistIn( nb_dims = d, batch_axes = [ num ] )
        sel = np.flatnonzero( ok )
        pts.x.set( moved[ sel ] if len( sel ) else np.zeros( ( 1, d ) ) )
        pts.facet.set( np.arange( len( sel ), dtype = np.int32 ) if len( sel ) else np.zeros( 1, np.int32 ) )
        pts.sign.set( np.ones( max( len( sel ), 1 ) ) )
        out = _DistOut( batch_axes = [ num ] )
        nfa = max( len( sel ), 1 )
        loom.ffi_call(
            "cell_complex_distances",
            FfiCode.per_item( code = f"sdot::cell_complex::distance_fwd<{ d }>( inputs.pts( batch_index ), inputs.area, inputs.centroid, "
                                     "outputs.out( batch_index ) );", includes = _INCLUDES ),
            FfiCode.per_item( f"sdot::cell_complex::distance_bwd<{ d }>( inputs.pts( batch_index ), inputs.area, inputs.centroid, "
                              "grad_of_outputs.out( batch_index ), grad_of_inputs.pts( batch_index ), grad_of_inputs.area, grad_of_inputs.centroid );",
                              computed_grads = "computed_grads.from_out_distance.pts.x = true; computed_grads.from_out_distance.area = true; "
                                               "computed_grads.from_out_distance.centroid = true;", includes = _INCLUDES ),
            pts = pts, out = loom.out( out ),
            area = RealTensor[ _ax( nfa, "num_lf" ), _ax( d, "dim" ) ]( area if len( sel ) else np.ones( ( 1, d ) ) ),
            centroid = RealTensor[ _ax( nfa, "num_lf" ), _ax( d, "dim" ) ]( cen if len( sel ) else np.zeros( ( 1, d ) ) ) )
        dist = _first( out.distance.raw, len( sel ) )
        if len( sel ) == m:
            return dist, facets
        if len( sel ) == 0:
            return _host_like( moved[ :, 0 ], np.full( m, np.nan ) ), facets
        return _scatter( dist, sel, m ) + _host_like( dist, np.where( ok, 0.0, np.nan ) ), facets

    # ---- helpers -------------------------------------------------------------------------------------

    def _local_facets( self, lfacets ):
        """`( areas, centroids )` of local facets ( dedup ids ), outward of their cell, in its frame"""
        cx, d = self.complex, self._d
        T = cx._tables()
        L = T[ "local_facets" ]
        corners, signs, owner = L( lfacets )
        nf = len( lfacets )
        s0 = np.searchsorted( owner, np.arange( nf ) ).astype( np.int32 )
        s1 = np.searchsorted( owner, np.arange( nf ), side = "right" ).astype( np.int32 )
        X = self._vertices()
        num_face = new_batch_axis( max( nf, 1 ), prefix = "face" )
        faces = _Face( batch_axes = [ num_face ] )
        faces.s0.set( s0 if nf else np.zeros( 1, np.int32 ) )
        faces.s1.set( s1 if nf else np.zeros( 1, np.int32 ) )
        out = _FaceOut( nb_dims = d, batch_axes = [ num_face ] )
        if len( corners ) == 0:
            corners, signs = np.zeros( ( 1, d ), np.int32 ), np.ones( 1 )
        k = d - 1
        loom.ffi_call(
            f"cell_complex_faces_{ k }",
            FfiCode.per_item( code = f"sdot::cell_complex::face_fwd<{ k },{ d }>( inputs.faces( batch_index ), inputs.X, inputs.corners, "
                                     "inputs.signs, inputs.apex_off, inputs.apex_rows, outputs.out( batch_index ) );", includes = _INCLUDES ),
            FfiCode.per_item( f"sdot::cell_complex::face_bwd<{ k },{ d }>( inputs.faces( batch_index ), inputs.X, inputs.corners, inputs.signs, "
                              "inputs.apex_off, inputs.apex_rows, outputs.out( batch_index ), grad_of_outputs.out( batch_index ), "
                              "grad_of_inputs.X );",
                              computed_grads = "computed_grads.from_out_measure.X = true; computed_grads.from_out_centroid.X = true; "
                                               "computed_grads.from_out_area.X = true;", includes = _INCLUDES ),
            faces = faces, out = loom.out( out ),
            corners = IntTensor[ _ax( len( corners ), "num_simplex" ), _ax( k + 1, "num_corner" ), dict( size = 32 ) ]( corners.astype( np.int32 ) ),
            signs = RealTensor[ _ax( len( corners ), "num_simplex" ) ]( signs.astype( float ) ),
            **self._apex_args( X ) )
        return _first( out.area.raw, nf ), _first( out.centroid.raw, nf )

    def _full( self, k ):
        """`( measures, centroids, areas )` over all the `k`-faces: the bounded ones from the kernel, the others infinite / NaN"""
        key = ( "full", k )
        if key in self._cache:
            return self._cache[ key ]
        cx, d, ops = self.complex, self._d, self._ops
        if k == 0:
            X = self.vertex_positions
            res = ( X[ :, 0 ] * 0 + 1, X, None )
        else:
            meas, cen, area = self._faces( k )
            bounded = np.zeros( cx.nb_faces( k ), bool )
            bounded[ cx._tables()[ "simplices" ][ k ][ "faces" ] ] = True
            if bounded.all():
                res = ( meas, cen, area )
            elif not bounded.any():
                m = cx.nb_faces( k )
                z = self.positions[ :1, 0 ] * 0
                res = ( z + _host_like( z, np.full( m, np.inf ) ), z[ :, None ] + _host_like( z, np.full( ( m, d ), np.nan ) ),
                        z[ :, None ] + _host_like( z, np.zeros( ( m, d ) ) ) )
            else:
                idx = np.flatnonzero( bounded )
                m = cx.nb_faces( k )
                # ( the others -- unbounded, or empty: a repeat of a computed one, then infinite ( zero for an empty cell ) or NaN --
                # additions of host constants )
                empty = cx._unbounded[ k ] == False
                res = ( _scatter( meas, idx, m ) * _host_like( meas, bounded.astype( float ) )
                        + _host_like( meas, np.where( bounded, 0.0, np.where( empty, 0.0, np.inf ) ) ),
                        _scatter( cen, idx, m ) + _host_like( cen, np.where( bounded, 0.0, np.nan )[ :, None ] ),
                        _scatter( area, idx, m ) * _host_like( area, bounded.astype( float )[ :, None ] ) )
        self._cache[ key ] = res
        return res


def _host_like( like, values ):
    """host `values` as an array of the framework, dtype and device of `like` ( a constant: nothing to derive )"""
    if isinstance( like, np.ndarray ):
        return np.asarray( values, dtype = like.dtype )
    ops = loom.ops( like )
    return ops.array( np.asarray( values ), ops.dtype_of( like ) )


def _scatter( values, idx, m ):
    """`values` put at rows `idx` of `m` rows ( the others: repeats of the first, to be masked ) -- by a gather, which every
    framework differentiates"""
    pos = np.zeros( m, np.int64 )
    pos[ idx ] = np.arange( len( idx ) )
    return values[ pos ]


def _affine( points, maps ):
    """`points [ m, d ]` ( any framework ) moved by per-point affine maps `[ m, d, d + 1 ]` ( host constants ), differentiably"""
    ops = loom.ops( points )
    d = maps.shape[ 1 ]
    if isinstance( points, np.ndarray ):
        return np.einsum( "mij,mj->mi", maps[ :, :, :d ], points ) + maps[ :, :, d ]
    dt = ops.dtype_of( points )
    M = ops.array( np.ascontiguousarray( maps[ :, :, :d ] ), dt )
    V = ops.array( np.ascontiguousarray( maps[ :, :, d ] ), dt )
    return ops.matmul( M, points[ :, :, None ] )[ :, :, 0 ] + V
