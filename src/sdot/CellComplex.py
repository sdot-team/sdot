"""THE CELL COMPLEX of a power diagram: its cells, and the faces they share -- vertices, edges, ..., facets -- each counted ONCE,
with what contains what.

`PowerDiagram.cells` gives each cell on its own: the facet between two seeds is there twice, a vertex as many times as cells
meet there. `PowerDiagram.cell_complex()` glues them: one numbering per dimension `k` ( `0` the vertices, `1` the edges, `d - 1`
the facets, `d` the cells, numbered like the seeds ), and the INCIDENCES between them -- the children of a `k`-face ( its
`k - 1`-faces ), its parents ( the `k + 1`-faces it bounds ), and more generally its `j`-faces for any `j`.

    cx = pd.cell_complex()
    for f in cx.facets:                          # each facet once
        a, b = f.sides                           # the seeds on both sides ( `b < 0`: a wall of the domain, `cx.walls`, `-1 - b` )
        f.measure, f.centroid, f.normal          # its area, its center of mass, its unit normal from `a` to `b`
        f.children                               # its edges in 3D, its vertices in 2D ( `Face`s )
    cx.cells[ 3 ].faces( 0 )                     # the vertices of the cell of seed 3
    cx.vertices[ 7 ].faces( cx.dim )             # the cells around vertex 7

Each dimension also reads AT ONCE ( `Faces` ): `cx.facets.sides`, `cx.edges.measures`, `cx.faces( k ).incident( j )` -- the
incidences as CSR arrays `( offsets, ids )`, the `j`-faces of face `i` being `ids[ offsets[ i ] : offsets[ i + 1 ] ]`.

Host side, in numpy: a snapshot of the diagram when it was asked for -- a new one after moving the seeds.

= Internals

THE GLUING. A vertex of cell `i` is the corner of `d` cuts; with the seed of the cell, the `d + 1` GENERATORS of the vertex ( seeds,
or walls of the domain ) name it in every cell that has it, as long as the diagram is in general position. Where it is not ( four
cells or more around a 2D vertex, a grid of seeds ), one point has several keys, which differ from cell to cell: the keys are
then merged by DISTANCE ( `tolerance`, on a grid of boxes: no quadratic search ) -- an edge shorter than that is not an edge. A face of dimension `k > 0` of a cell is the set of the vertices on `d - k` of its cuts ( a simple
polytope: each `d - k` cuts of a vertex carry a face through it ), and it is named, across cells, by its set of GLOBAL vertices --
a face that collapsed to fewer than `k + 1` vertices ( a tiny edge merged ) is not a face. An unbounded face ( on a vertex
of an `INFINITE` wall, made up ) is named by its finite vertices AND its generators.

The walls of the domain all carry `BOUNDARY` in the cells: each is told apart by the half-space of `walls` its vertices lie on.
"""

import itertools
import math

import numpy as np

from .Cell import INFINITE, SUPPORT
from loom.tensor import Axis, IntTensor, RealTensor, ShapeVar
from .viz.Displayable import Displayable


class CellComplex:
    """The cells of a diagram glued along their faces: every face once, numbered per dimension, with its incidences. Built by
    `PowerDiagram.cell_complex()` -- or from any batch of cells, `CellComplex( cells, walls )`, the items being the seeds their
    cut identifiers designate.

    `faces( k )` the `k`-faces, `vertices` / `edges` / `facets` / `cells` those of dimension `0`, `1`, `d - 1`, `d`. `walls`: the
    half-spaces `( directions, offsets )` of the domain, which the negative sides of the facets designate ( `-1 - wall` ).
    `tolerance`: the distance under which two vertices are one, where the diagram is not in general position ( a grid of seeds ).

    = Internals

    `seeds` ( `[ n, d ]`, in the frame of the cells ) and `weights`: host values, for `locate` ( when the cells are their power cells,
    `power`: not with groups ) and the choices of the geometry. `values`: `( positions, weights )` as the framework holds them ( the given
    positions ), what `geometry()` differentiates by default; `fold`: per seed, the affine map from them to the frame of the cuts. `group` ( a `periodicity.PeriodicSetup` ): the
    cells are those of a quotient, their cut identifiers `j + n t` the image `t` of
    seed `j`, and the faces are glued MODULO the group. `maps` ( `[ n, d, d + 1 ]` ): the frame each cell is read in by the user
    ( the given positions of a periodic diagram ), the cells being given in the frame where they were cut ( the folded seeds )."""

    def __init__( self, cells, walls = None, tolerance = None, seeds = None, weights = None, power = True, group = None, maps = None,
                  values = None, fold = None ):
        self._d = d = int( cells.dim )
        self._n = n = int( cells.nb_items )
        W, vcell, VC, CID, ccell, coff = _bulk( cells )   # ( `W`: the frame of the cuts, where the gluing happens )
        V = W if maps is None else _moved( W, vcell, maps )
        self._W, self._V, self._vcell, self._cid = W, V, vcell, CID
        self._seeds = None if seeds is None else np.asarray( seeds, dtype = float ).reshape( n, d )
        self._weights = np.zeros( n ) if weights is None else np.asarray( weights, dtype = float ).reshape( n )
        self._group, self._maps = group, None if maps is None else np.asarray( maps, dtype = float )
        self.periodic = group is not None
        self._power = bool( power )
        self._values = values                          # ( positions, weights ) of the framework: what `geometry()` reads by default
        self._fold = None if fold is None else np.asarray( fold, dtype = float )   # per seed, from the given positions to the frame of the cuts

        # the scale, and the tolerance of the merge ( in the kernel's float, which is what the vertices are exact to )
        fake_cut = CID == INFINITE
        fake_v = fake_cut[ VC ].any( axis = 1 ) if len( VC ) else np.zeros( 0, bool )
        real = W[ ~fake_v ]
        scale = float( np.ptp( real, axis = 0 ).max() ) if len( real ) else 1.0
        if tolerance is None:
            tolerance = ( 1e-12 if str( getattr( cells, "kernel_dtype", "FP32" ) ) == "FP64" else 1e-5 ) * max( scale, 1e-300 )
        self.tolerance = tol = float( tolerance )

        # THE GENERATORS of each cut: the seed it faces ( an image of it with a periodicity ), or the wall it lies on ( `-1 - wall` )
        gen = CID.astype( np.int64 )
        dom = ( CID < 0 ) & ( CID > SUPPORT )
        if dom.any():
            if walls is None:
                walls = _fit_walls( W, vcell, VC, dom, tol )
            gen[ dom ] = -1 - _wall_of_cuts( W, VC, dom, walls )
        # ... and with a periodicity, the image: the cut identifier `j + n t` faces `maps[ t ]( seed j )` ( `keep_images` of the cells )
        img = np.zeros( len( CID ), np.int64 )
        if group is not None:
            img[ gen >= 0 ] = gen[ gen >= 0 ] // max( n, 1 )
            gen[ gen >= 0 ] %= max( n, 1 )
        self.walls = None if walls is None else ( np.asarray( walls[ 0 ], dtype = float ).reshape( -1, d ),
                                                  np.asarray( walls[ 1 ], dtype = float ).reshape( -1 ) )
        self._gen, self._img = gen, img
        quotient = None if group is None else _Quotient( group, tol )

        # THE VERTICES: named EXACTLY by their generators ( the seed of the cell and the `d` cuts, seeds with their images or walls ),
        # in the frame of the smallest seed among them -- the same from every cell that has the vertex --, then merged by distance
        # where the diagram is degenerate. `E`: per local vertex, the element of the group from that frame to the frame of its cell
        gv, self._E = _glue_vertices( W, vcell, VC, gen, img, fake_v, tol, quotient, self.walls )
        # THE GLUING, CHECKED: a facet that only one of its two cells has ( the cells disagree ) is contracted to a point on both
        # sides when it is tiny, and the faces glued again ( `_check` )
        self._ctx = dict( W = W, V = V, vcell = vcell, VC = VC, ccell = ccell, fake_cut = fake_cut, fake_v = fake_v, gen = gen,
                          quotient = quotient, scale = scale )
        self._diag = dict( contracted_facets = 0, one_sided_facets = 0, open_links = 0 )
        for it in range( 8 ):
            self._build( gv )
            merges = self._check( final = it == 7 )
            if merges is None:
                break
            gv = _merge_ids( gv, merges )
            self._diag[ "contracted_facets" ] += len( merges )

    def _build( self, gv ):
        """the faces of the complex, from the global numbers of the local vertices `gv` ( the vertex gluing ): the true face lattice of
        each cell and the global names of its faces, by `sdot/complex/Glue.h`"""
        d, n = self._d, self._n
        V, vcell = self._ctx[ "V" ], self._ctx[ "vcell" ]
        quotient = self._ctx[ "quotient" ]
        self._gv0 = gv
        out = self._glue_kernel( gv )
        f_cell, f_dim, f_gid, f_cut, f_fake, f_off, f_pts = ( out[ k ] for k in ( "f_cell", "f_dim", "f_gid", "f_cut", "f_fake", "f_off", "f_pts" ) )
        nbk = out[ "nb" ]

        # THE VERTICES: the points that are corners, numbered by the gluing ( the local vertices merged into them are not rows )
        d0 = np.flatnonzero( ( f_dim == 0 ) & ( f_gid >= 0 ) & ( np.diff( f_off ) == 1 ) & ( f_fake == 0 ) )
        reps0 = f_pts[ f_off[ d0 ] ]
        self._gv = np.full( len( V ), -1, np.int64 )
        self._gv[ reps0 ] = f_gid[ d0 ]
        nv = int( nbk[ 0 ] )
        self._vertex_positions = np.zeros( ( nv, d ) )
        if nv:
            if quotient is None:                         # ( the mean of its copies, all at the same place )
                np.add.at( self._vertex_positions, f_gid[ d0 ], V[ reps0 ] )
                self._vertex_positions /= np.maximum( np.bincount( f_gid[ d0 ], minlength = nv ), 1 )[ :, None ]
            else:                                        # ( one of its copies: they are a period apart )
                self._vertex_positions[ f_gid[ d0[ ::-1 ] ] ] = V[ reps0[ ::-1 ] ]
        # the equations of each row: the cuts of `D` facets of its vertex, independent
        self._veq = np.full( ( len( V ), d ), -1, np.int64 )
        eqi = np.cumsum( f_dim == 0 ) - 1                # ( the equations come in the order of the faces of dimension 0 )
        self._veq[ reps0 ] = out[ "v_eq" ][ eqi[ d0 ] ]

        # THE LOCAL FACES, codimension by codimension ( `c = 0` the cells, `c = d` the vertices ), and their global numbers
        self._nb, self._unbounded, self._loc = {}, {}, {}
        idx_of = np.zeros( len( f_dim ), np.int64 )
        cnt = np.diff( f_off )
        for c in range( d + 1 ):
            k = d - c
            sel = np.flatnonzero( f_dim == k )
            idx_of[ sel ] = np.arange( len( sel ) )
            lf = np.repeat( np.arange( len( sel ) ), cnt[ sel ] )
            starts = np.repeat( f_off[ sel ], cnt[ sel ] )
            lv = f_pts[ starts + np.arange( len( lf ) ) - np.repeat( np.cumsum( cnt[ sel ] ) - cnt[ sel ], cnt[ sel ] ) ] if len( lf ) else np.zeros( 0, np.int64 )
            lg = f_gid[ sel ].astype( np.int64 )
            fake = f_fake[ sel ].astype( bool )
            self._nb[ k ] = int( nbk[ k ] )
            unb = np.zeros( self._nb[ k ], bool )
            unb[ lg[ fake & ( lg >= 0 ) ] ] = True
            self._unbounded[ k ] = unb
            self._loc[ c ] = dict( keys = f_cut[ sel ].astype( np.int64 )[ :, None ], lg = lg, cell = f_cell[ sel ].astype( np.int64 ),
                                   lf = lf, lv = lv.astype( np.int64 ), fake = fake, lf0 = lf, lv0 = lv.astype( np.int64 ) )
        self._nb[ d ] = n
        self._lchild = {}
        cp, cc = out[ "c_par" ], out[ "c_chi" ]
        pc = d - f_dim[ cp ]
        for c in range( d ):
            m = pc == c
            self._lchild[ c ] = ( idx_of[ cp[ m ] ], idx_of[ cc[ m ] ] )

        # ... and the global incidences: the children, the vertices and the cells of each face
        gvn = self._gv
        self._pairs = {}
        for c in range( d ):
            par, chi = self._lchild[ c ]
            gp, gc = self._loc[ c ][ "lg" ][ par ], self._loc[ c + 1 ][ "lg" ][ chi ]
            ok = ( gp >= 0 ) & ( gc >= 0 )
            self._pairs[ ( d - c, d - c - 1 ) ] = _unique_pairs( gp[ ok ], gc[ ok ] )
        for c in range( d ):
            L = self._loc[ c ]
            g, v = L[ "lg" ][ L[ "lf" ] ], gvn[ L[ "lv" ] ]
            ok = ( g >= 0 ) & ( v >= 0 )
            self._pairs[ ( d - c, 0 ) ] = _unique_pairs( g[ ok ], v[ ok ] )
        for c in range( 1, d + 1 ):
            L = self._loc[ c ]
            ok = L[ "lg" ] >= 0
            self._pairs[ ( d - c, d ) ] = _unique_pairs( L[ "lg" ][ ok ], L[ "cell" ][ ok ] )

        # the REPRESENTATIVE of each face: one of its local copies, which its geometry is read off
        self._rep = {}
        for c in range( d + 1 ):
            lg = self._loc[ c ][ "lg" ]
            ok = np.flatnonzero( lg >= 0 )
            rep = np.full( self._nb[ d - c ], -1, np.int64 )
            rep[ lg[ ok[ ::-1 ] ] ] = ok[ ::-1 ]
            self._rep[ d - c ] = rep
        self._csr, self._geom = {}, {}

    def _cut_planes( self ):
        """`[ nb_cuts, d + 1 ]`: the exact plane of each cut in the frame of its cell, unit normal and offset -- the bisector of the seed
        of the cell and the image of the seed it faces, a wall of the domain; read off the vertices for a made-up wall ( or cells without
        seeds )"""
        if "planes" in self._ctx:
            return self._ctx[ "planes" ]
        d, n = self._d, self._n
        W, VC, ccell, CID = self._ctx[ "W" ], self._ctx[ "VC" ], self._ctx[ "ccell" ], self._cid
        gen, img = self._gen, self._img
        nc = len( CID )
        planes = np.zeros( ( nc, d + 1 ) )
        done = np.zeros( nc, bool )
        if self._seeds is not None:
            S, w = self._seeds, self._weights
            eye = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
            G = self._group.maps if self._group is not None else eye[ None ]
            sd = np.flatnonzero( ( gen >= 0 ) & ( CID != INFINITE ) )
            ps = S[ ccell[ sd ] ]
            Gt = G[ img[ sd ] ]
            q = np.einsum( "mij,mj->mi", Gt[ :, :, :d ], S[ gen[ sd ] ] ) + Gt[ :, :, d ]
            nrm = q - ps
            off = ( ( q ** 2 ).sum( axis = 1 ) - ( ps ** 2 ).sum( axis = 1 ) - w[ gen[ sd ] ] + w[ ccell[ sd ] ] ) / 2
            ln = np.maximum( np.linalg.norm( nrm, axis = 1 ), 1e-300 )
            planes[ sd, :d ], planes[ sd, d ] = nrm / ln[ :, None ], off / ln
            done[ sd ] = True
            if self.walls is not None and len( self.walls[ 1 ] ):
                wd = np.flatnonzero( ( gen < 0 ) & ( CID < 0 ) & ( CID > SUPPORT ) )
                u, o = _unit_planes( *self.walls )
                planes[ wd, :d ], planes[ wd, d ] = u[ -1 - gen[ wd ] ], o[ -1 - gen[ wd ] ]
                done[ wd ] = True
        rest = np.flatnonzero( ~done )
        if len( rest ):                                  # ( from the vertices: the plane they span, outward of their cell )
            vrep = np.repeat( np.arange( len( VC ) ), d )
            pr = _unique_pairs( VC.reshape( -1 ), vrep )
            pr = pr[ np.isin( pr[ :, 0 ], rest ) ]
            off, ids = _csr( pr, nc )
            B = _bases( W, off, ids, d )
            nrm = B[ :, d - 1 ]
            ctr = W[ ids[ np.minimum( off[ :-1 ], max( len( ids ) - 1, 0 ) ) ] ] if len( ids ) else np.zeros( ( nc, d ) )
            inside = np.zeros( ( n, d ) )
            np.add.at( inside, self._ctx[ "vcell" ], W )
            inside /= np.maximum( np.bincount( self._ctx[ "vcell" ], minlength = n ), 1 )[ :, None ]
            flip = ( ( inside[ ccell ] - ctr ) * nrm ).sum( axis = 1 ) > 0
            nrm[ flip ] *= -1
            planes[ rest, :d ] = nrm[ rest ]
            planes[ rest, d ] = ( nrm * ctr ).sum( axis = 1 )[ rest ]
        self._ctx[ "planes" ] = planes
        return planes

    def _glue_kernel( self, gv ):
        """the outputs of `sdot::glue::run` ( see `Glue.h` ), with room enough"""
        from loom.compilation.FfiCode import FfiCode
        import loom
        d, n = self._d, self._n
        W, VC, vcell, ccell = self._ctx[ "W" ], self._ctx[ "VC" ], self._ctx[ "vcell" ], self._ctx[ "ccell" ]
        planes = self._cut_planes()
        nV, nC = len( W ), len( planes )
        eye = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
        G = self._group.maps if self._group is not None else eye[ None ]
        walls = np.concatenate( [ *_unit_planes( *self.walls )[ :1 ], _unit_planes( *self.walls )[ 1 ][ :, None ] ], axis = 1 ) \
            if self.walls is not None and len( self.walls[ 1 ] ) else np.zeros( ( 1, d + 1 ) )
        E = self._E if self._group is not None else np.broadcast_to( eye, ( max( nV, 1 ), d, d + 1 ) )
        cell_v = np.concatenate( [ [ 0 ], np.cumsum( np.bincount( vcell, minlength = n ) ) ] )
        cell_c = np.concatenate( [ [ 0 ], np.cumsum( np.bincount( ccell, minlength = n ) ) ] )
        # ( the resolution of the lattice is that of the vertex gluing: two distinct vertices are more than `tol` apart, so an edge
        # between them is one; the copies of one vertex in a cell are within `_same_place` )
        params = np.array( [ self.tolerance, _same_place( self.tolerance, d ), float( self._group is not None ) ] )
        ax = lambda m, name: Axis( ShapeVar( int( max( m, 1 ) ) ), name = name )
        I32 = lambda a, *axes: IntTensor[ ( *axes, dict( size = 32 ) ) ]( np.ascontiguousarray( a, dtype = np.int32 ) )
        nv_ax, nc_ax, dim = ax( nV, "num_lv" ), ax( nC, "num_cut" ), ax( d, "dim" )
        caps = self._ctx.get( "glue_caps" ) or [ max( 16, 4 * nV * d ), max( 16, 16 * nV * d ), max( 16, 8 * nV * d ), max( 16, nV ) ]
        while True:
            cf, cp, cch, cv = caps
            outs = dict( f_cell = IntTensor[ ax( cf, "cap_f" ), dict( size = 32 ) ](), f_dim = IntTensor[ ax( cf, "cap_f" ), dict( size = 32 ) ](),
                         f_gid = IntTensor[ ax( cf, "cap_f" ), dict( size = 32 ) ](), f_cut = IntTensor[ ax( cf, "cap_f" ), dict( size = 32 ) ](),
                         f_fake = IntTensor[ ax( cf, "cap_f" ), dict( size = 32 ) ](), f_off = IntTensor[ ax( cf + 1, "cap_f1" ), dict( size = 32 ) ](),
                         f_pts = IntTensor[ ax( cp, "cap_p" ), dict( size = 32 ) ](), c_par = IntTensor[ ax( cch, "cap_c" ), dict( size = 32 ) ](),
                         c_chi = IntTensor[ ax( cch, "cap_c" ), dict( size = 32 ) ](), v_eq = IntTensor[ ax( cv, "cap_v" ), dim, dict( size = 32 ) ](),
                         stats = IntTensor[ ax( 4 + d + 1, "num_stat" ), dict( size = 64 ) ]() )
            loom.ffi_call(
                f"cell_complex_glue_{ d }",
                FfiCode.inline( f"sdot::glue::run<{ d }>( queue, args.inputs.W, args.inputs.VC, args.inputs.gv, args.inputs.E, args.inputs.planes, "
                                "args.inputs.cut_inf, args.inputs.cgen, args.inputs.cimg, args.inputs.cell_v, args.inputs.cell_c, args.inputs.G, "
                                "args.inputs.walls, args.inputs.params, args.outputs.f_cell, args.outputs.f_dim, args.outputs.f_gid, "
                                "args.outputs.f_cut, args.outputs.f_fake, args.outputs.f_off, args.outputs.f_pts, args.outputs.c_par, "
                                "args.outputs.c_chi, args.outputs.v_eq, args.outputs.stats );", includes = [ "sdot/complex/Glue.h" ] ),
                W = RealTensor[ nv_ax, dim ]( W if nV else np.zeros( ( 1, d ) ) ),
                VC = I32( VC if nV else np.zeros( ( 1, d ) ), nv_ax, dim ),
                gv = I32( gv if nV else np.full( 1, -1 ), nv_ax ),
                E = RealTensor[ nv_ax, dim, ax( d + 1, "aff" ) ]( np.ascontiguousarray( E[ :max( nV, 1 ) ] ) ),
                planes = RealTensor[ nc_ax, ax( d + 1, "aff" ) ]( planes if nC else np.zeros( ( 1, d + 1 ) ) ),
                cut_inf = I32( ( self._cid == INFINITE ) if nC else np.zeros( 1 ), nc_ax ),
                cgen = I32( self._gen if nC else np.zeros( 1 ), nc_ax ),
                cimg = I32( self._img if nC else np.zeros( 1 ), nc_ax ),
                cell_v = I32( cell_v, ax( n + 1, "num_cell1" ) ),
                cell_c = I32( cell_c, ax( n + 1, "num_cell1" ) ),
                G = RealTensor[ ax( len( G ), "num_map" ), dim, ax( d + 1, "aff" ) ]( G ),
                walls = RealTensor[ ax( len( walls ), "num_wall" ), ax( d + 1, "aff" ) ]( walls ),
                params = RealTensor[ ax( 3, "num_param" ) ]( params ),
                **{ k: loom.out( v ) for k, v in outs.items() },
                has_dynamic_capacity = False,
            )
            st = np.asarray( outs[ "stats" ] ).reshape( -1 ).astype( np.int64 )
            need = [ int( st[ 0 ] ), int( st[ 1 ] ), int( st[ 2 ] ), int( st[ 3 ] ) ]
            if all( a <= b for a, b in zip( need, caps ) ):
                break
            caps = [ max( a, b ) for a, b in zip( need, caps ) ]
        self._ctx[ "glue_caps" ] = caps
        nf, np_, nch, nvv = need
        g = lambda k, m: np.asarray( outs[ k ] ).reshape( -1 ).astype( np.int64 )[ :m ]
        return dict( f_cell = g( "f_cell", nf ), f_dim = g( "f_dim", nf ), f_gid = g( "f_gid", nf ), f_cut = g( "f_cut", nf ),
                     f_fake = g( "f_fake", nf ), f_off = g( "f_off", nf + 1 ), f_pts = g( "f_pts", np_ ),
                     c_par = g( "c_par", nch ), c_chi = g( "c_chi", nch ),
                     v_eq = np.asarray( outs[ "v_eq" ] ).reshape( -1, d ).astype( np.int64 )[ :nvv ], nb = st[ 4: ] )

    # ---- what a user reads ------------------------------------------------------------------------

    @property
    def dim( self ):
        """The dimension of the space ( and of the cells )."""
        return self._d

    def nb_faces( self, k ):
        """The number of `k`-faces."""
        return int( self._nb[ self._check_dim( k ) ] )

    def faces( self, k ) -> "Faces":
        """The `k`-faces, `0 <= k <= dim`: a sequence of `Face`, and their arrays ( `Faces` )."""
        return Faces( self, self._check_dim( k ) )

    @property
    def vertices( self ) -> "Faces":
        """The `0`-faces."""
        return self.faces( 0 )

    @property
    def edges( self ) -> "Faces":
        """The `1`-faces ( the cells themselves in 1D )."""
        return self.faces( 1 )

    @property
    def facets( self ) -> "Faces":
        """The `( dim - 1 )`-faces: what two cells share, or a cell and a wall of the domain ( `Faces.sides` )."""
        return self.faces( self._d - 1 )

    @property
    def cells( self ) -> "Faces":
        """The `dim`-faces, numbered like the seeds ( an empty cell has no face )."""
        return self.faces( self._d )

    @property
    def vertex_positions( self ):
        """`[ nb_vertices, d ]`: the position of each vertex ( with a periodicity, of one of its copies ) -- a tensor of the framework of
        the positions, differentiable ( `geometry` )."""
        return self.geometry().vertex_positions

    def incident( self, k, j ):
        """The `j`-faces of each `k`-face, as a CSR `( offsets, ids )`: those of face `i` are `ids[ offsets[ i ] : offsets[ i + 1 ] ]`
        -- the faces it contains for `j < k`, the faces that contain it for `j > k`, sorted, except the vertices of a `2`-face, in
        the order of its boundary ( counterclockwise for a 2D cell, around `normal` for a 3D facet; with a periodicity, a vertex
        can come back there, two of its copies bounding the face ). `j == k`: each face itself."""
        k, j = self._check_dim( k ), self._check_dim( j )
        if ( k, j ) not in self._csr:
            if j == k:
                m = self.nb_faces( k )
                res = ( np.arange( m + 1 ), np.arange( m ) )
            elif k == 2 and j == 0:
                off, lv = self._rep_points( 2 )
                res = ( off, self._gv[ lv ] )
            else:
                res = _csr( self._pairs_of( k, j ), self.nb_faces( k ) )
            self._csr[ ( k, j ) ] = res
        return self._csr[ ( k, j ) ]

    def __repr__( self ):
        return f"CellComplex( dim = { self._d }, " + ", ".join( f"{ self.nb_faces( k ) } { _NAMES.get( ( self._d, k ), f'{ k }-faces' ) }"
                                                                for k in range( self._d + 1 ) ) + ( ", periodic )" if self.periodic else " )" )

    # ---- the check -----------------------------------------------------------------------------------

    def _check( self, final = False ):
        """THE CONSISTENCY OF THE GLUING, into `diagnostics`, and the vertices to merge ( pairs of global numbers ) or `None`:

          * a facet between two seeds that only one of them has: contracted to a point when it is smaller than `100 tolerance`
            ( its vertices merged, the faces glued again ), counted in `one_sided_facets` otherwise;
          * `facet_mismatch`: the largest difference between the measures of the two copies of a facet, relative to the facet of a
            typical cell ( `( volume / n )^( ( d - 1 ) / d )` ) ( what the cells disagree on, without changing the connectivity );
          * `open_links`: the faces of dimension `d - 2` around which the facets do not close ( as many facets as cells inside the
            domain, one more on its boundary ) -- what a cell that misses a facet leaves."""
        d = self._d
        if d < 2:
            self._diag.update( one_sided_facets = 0, open_links = 0 )
            return None
        sides = self._sides_and_normals()[ 0 ]
        off, _ = self.incident( d - 1, d )
        cnt = np.diff( off )
        inner = ( sides[ :, 1 ] >= 0 ) & ( sides[ :, 0 ] != sides[ :, 1 ] )
        one = np.flatnonzero( inner & ( cnt == 1 ) )

        # the one-sided facets: their extent, from the vertices of their copy
        ro, rv = self._rep_points( d - 1 ) if len( one ) else ( None, None )
        V, gv = self._V, self._gv0                       # ( the merges are those of the vertex gluing )
        merges, big = [], 0
        for f in one:
            lv = rv[ ro[ f ] : ro[ f + 1 ] ]
            if len( lv ) == 0:
                continue
            P = V[ lv ]
            if np.abs( P.max( axis = 0 ) - P.min( axis = 0 ) ).max() <= 100 * self.tolerance:
                g = np.unique( gv[ lv ] )
                merges += [ ( int( g[ 0 ] ), int( h ) ) for h in g[ 1: ] ]
            else:
                big += 1
        if merges and not final:
            return merges
        big += len( merges ) if final else 0

        # the links of the faces of dimension d - 2
        po, pf = self.incident( d - 2, d - 1 )
        nf = np.diff( po )
        nc = np.diff( self.incident( d - 2, d )[ 0 ] )
        wall = np.zeros( len( nf ), bool )
        owner = np.repeat( np.arange( len( nf ) ), nf )
        wall[ owner[ sides[ pf, 1 ] < 0 ] ] = True
        bounded = ~self._unbounded[ d - 2 ]
        closed = np.where( wall, nf == nc + 1, nf == nc ) | ~bounded
        if self.periodic:                                # ( a cell around a face twice, two of its copies: counted once )
            closed |= nf >= nc
        self._diag.update( one_sided_facets = big, open_links = int( ( ~closed ).sum() ) )
        if big or not closed.all():
            import warnings
            warnings.warn( f"cell_complex: the cells disagree on { big } facets larger than 100 x tolerance, and { int( ( ~closed ).sum() ) } "
                           f"faces of dimension { d - 2 } are not closed -- see `CellComplex.diagnostics`", stacklevel = 3 )
        return None

    @property
    def diagnostics( self ):
        """What the check of the gluing found ( see `_check` ): `contracted_facets`, `one_sided_facets`, `open_links`, and
        `facet_mismatch` ( computed when read: the measures of the copies of every facet )."""
        if "facet_mismatch" not in self._diag:
            self._diag[ "facet_mismatch" ] = self._facet_mismatch()
        return self._diag

    def _facet_mismatch( self ):
        """the largest difference between the measures of the copies of a facet, relative to the facet of a typical cell"""
        d = self._d
        if d < 2:
            return 0.0
        sides = self._sides_and_normals()[ 0 ]
        off, _ = self.incident( d - 1, d )
        cnt = np.diff( off )
        inner = ( sides[ :, 1 ] >= 0 ) & ( sides[ :, 0 ] != sides[ :, 1 ] )
        # the two copies of each facet
        meas = self._local_geom( 1 )[ 0 ]
        lg = self._loc[ 1 ][ "lg" ]
        ok = lg >= 0
        hi = np.full( self.nb_faces( d - 1 ), -np.inf )
        lo = np.full( self.nb_faces( d - 1 ), np.inf )
        np.maximum.at( hi, lg[ ok ], meas[ ok ] )
        np.minimum.at( lo, lg[ ok ], meas[ ok ] )
        two = inner & ( cnt == 2 ) & np.isfinite( hi )
        cm = self._local_geom( 0 )[ 0 ]                  # ( relative to the facet of a typical cell, not to the slivers )
        cm = cm[ np.isfinite( cm ) & ( cm > 0 ) ]
        ref = float( np.mean( cm ) ) ** ( ( d - 1 ) / d ) if len( cm ) else 1.0
        return float( ( hi[ two ] - lo[ two ] ).max() / ref ) if two.any() else 0.0


    @property
    def is_consistent( self ):
        """Whether every facet between two seeds is in both their cells and the facets close around every face of dimension `d - 2`
        ( see `diagnostics`: `one_sided_facets`, `open_links`, `contracted_facets`, `facet_mismatch` )."""
        return self._diag.get( "one_sided_facets", 0 ) == 0 and self._diag.get( "open_links", 0 ) == 0

    # ---- incidences ------------------------------------------------------------------------------

    def _check_dim( self, k ):
        k = int( k )
        if not 0 <= k <= self._d:
            raise ValueError( f"a face of dimension { k }: 0 .. { self._d } in this complex" )
        return k

    def _pairs_of( self, k, j ):
        """`[ m, 2 ]` ( k-face, j-face ): found while gluing, or composed from the children / parents"""
        if ( k, j ) in self._pairs:
            return self._pairs[ ( k, j ) ]
        if j > k:
            res = self._pairs_of( j, k )[ :, ::-1 ].copy()
        else:
            res = _compose( self._pairs_of( k, k - 1 ), self._pairs_of( k - 1, j ), self.nb_faces( k - 1 ) )
        self._pairs[ ( k, j ) ] = res
        return res

    # ---- the local copies --------------------------------------------------------------------------

    def _local_csr( self, c ):
        """the local vertices of each local face of codimension `c`, `( offsets, flat vertex ids )`"""
        key = ( "lcsr", c )
        if key not in self._geom:
            L = self._loc[ c ]
            self._geom[ key ] = _csr( np.stack( [ L[ "lf0" ], L[ "lv0" ] ], axis = 1 ), len( L[ "keys" ] ) )
        return self._geom[ key ]

    def _local_geom( self, c ):
        """`( measures, centroids )` of the local faces of codimension `c`, from those of their children: a face is the union of
        the pyramids of apex a point inside it ( the mean of its vertices ) over its faces one dimension below -- height times base
        over `k`, and a centroid at `( apex + k base centroid ) / ( k + 1 )`. In the frame of the user. Infinite ( and no centroid )
        for an unbounded face."""
        key = ( "lgeom", c )
        if key in self._geom:
            return self._geom[ key ]
        d, V, L = self._d, self._V, self._loc[ c ]
        k, m = d - c, len( L[ "keys" ] )
        off, ids = self._local_csr( c )
        if k == 0:
            res = ( np.ones( m ), V[ ids[ np.minimum( off[ :-1 ], max( len( ids ) - 1, 0 ) ) ] ] if m and len( ids ) else np.zeros( ( m, d ) ) )   # ( a made-up vertex has no point )
        else:
            owner = np.repeat( np.arange( m ), np.diff( off ) )
            apex = np.zeros( ( m, d ) )
            np.add.at( apex, owner, V[ ids ] )
            apex /= np.maximum( np.diff( off ), 1 )[ :, None ]
            f, g = self._lchild[ c ]
            cm, cc = self._local_geom( c + 1 )
            goff, gids = self._local_csr( c + 1 )
            # the height of the apex over the affine hull of the child, inside the face
            r = apex[ f ] - V[ gids[ goff[ g ] ] ] if len( g ) else np.zeros( ( 0, d ) )
            if k >= 2 and len( g ):
                B = _bases( V, goff, gids, k - 1 )[ g ]
                r = r - np.einsum( "pq,pqd->pd", np.einsum( "pqd,pd->pq", B, r ), B )
            h = np.linalg.norm( r, axis = 1 )
            with np.errstate( invalid = "ignore", divide = "ignore" ):      # ( an unbounded child: set below )
                w = h * cm[ g ] / k
                meas = np.bincount( f, weights = w, minlength = m )
                cen = np.zeros( ( m, d ) )
                np.add.at( cen, f, w[ :, None ] * ( apex[ f ] + k * cc[ g ] ) / ( k + 1 ) )
                cen /= meas[ :, None ]
            unb = L[ "fake" ].copy()
            if len( f ):
                unb[ f[ ~np.isfinite( cm[ g ] ) ] ] = True
            meas[ unb ] = np.inf
            cen[ unb ] = np.nan
            res = ( meas, cen )
        self._geom[ key ] = res
        return res

    def _rep_points( self, k ):
        """`( offsets, flat vertex ids )`: the vertices of the representative of each `k`-face -- without the made-up ones nor the
        copies of one vertex that the merge says are one, in the order of the boundary for a `2`-face ( oriented as `incident`
        says ), by global number otherwise"""
        key = ( "rep", k )
        if key in self._geom:
            return self._geom[ key ]
        d, c = self._d, self._d - k
        rep = self._rep[ k ]
        off, ids = self._local_csr( c )
        has = rep >= 0                                   # ( an empty cell has no copy )
        cnt = np.where( has, np.diff( off )[ np.maximum( rep, 0 ) ], 0 )
        owner = np.repeat( np.arange( len( rep ) ), cnt )
        lv = ids[ np.repeat( off[ np.maximum( rep, 0 ) ], cnt ) + np.arange( len( owner ) ) - np.repeat( np.cumsum( cnt ) - cnt, cnt ) ]
        keep = ( self._gv[ lv ] >= 0 )
        owner, lv = owner[ keep ], lv[ keep ]
        V = self._V
        if k == 2 and len( lv ):
            o2 = np.concatenate( [ [ 0 ], np.cumsum( np.bincount( owner, minlength = len( rep ) ) ) ] )
            basis = _bases( V, o2, lv, 2 )
            if d == 3:                                   # around the normal of the facet, from its first side to its second
                flip = ( np.cross( basis[ :, 0 ], basis[ :, 1 ] ) * self._sides_and_normals()[ 1 ] ).sum( axis = 1 ) < 0
            elif d == 2:                                 # counterclockwise
                flip = basis[ :, 0, 0 ] * basis[ :, 1, 1 ] - basis[ :, 0, 1 ] * basis[ :, 1, 0 ] < 0
            else:
                flip = np.zeros( len( rep ), bool )
            basis[ flip, 1 ] *= -1
            mean = np.zeros( ( len( rep ), d ) )
            np.add.at( mean, owner, V[ lv ] )
            mean /= np.maximum( np.diff( o2 ), 1 )[ :, None ]
            q = V[ lv ] - mean[ owner ]
            ang = np.arctan2( ( q * basis[ owner, 1 ] ).sum( axis = 1 ), ( q * basis[ owner, 0 ] ).sum( axis = 1 ) )
            order = np.lexsort( ( ang, owner ) )
            owner, lv = owner[ order ], lv[ order ]
            # the copies of one vertex, one after the other on the boundary ( a merged edge ): once
            nxt = np.arange( len( lv ) ) + 1
            last = np.r_[ owner[ 1: ] != owner[ :-1 ], True ]
            nxt[ last ] = o2[ owner[ last ] ]
            same = ( self._gv[ lv ] == self._gv[ lv[ nxt ] ] ) & ( np.abs( V[ lv ] - V[ lv[ nxt ] ] ).max( axis = 1 ) <= _same_place( self.tolerance, d ) )
            same &= nxt != np.arange( len( lv ) )
            owner, lv = owner[ ~same ], lv[ ~same ]
        elif len( lv ):
            order = np.lexsort( ( self._gv[ lv ], owner ) )
            owner, lv = owner[ order ], lv[ order ]
            dup = np.r_[ False, ( owner[ 1: ] == owner[ :-1 ] ) & ( self._gv[ lv[ 1: ] ] == self._gv[ lv[ :-1 ] ] )
                         & ( np.abs( V[ lv[ 1: ] ] - V[ lv[ :-1 ] ] ).max( axis = 1 ) <= _same_place( self.tolerance, d ) ) ]
            owner, lv = owner[ ~dup ], lv[ ~dup ]
        res = ( np.concatenate( [ [ 0 ], np.cumsum( np.bincount( owner, minlength = len( rep ) ) ) ] ), lv )
        self._geom[ key ] = res
        return res

    def _local_polygons( self, k, i ):
        """the polygons that draw the `k`-face `i` ( `k >= 2` ), as lists of positions: its representative ( `k == 2` ), or the
        `2`-faces of its representative, in its own frame ( `k == 3` )"""
        if k == 2:
            off, lv = self._rep_points( 2 )
            return [ self._V[ lv[ off[ i ] : off[ i + 1 ] ] ] ]
        c = self._d - k
        par, chi = self._lchild[ c ]
        res = []
        off, ids = self._local_csr( c + 1 )
        fake = self._loc[ c + 1 ][ "fake" ]
        for lf in chi[ par == self._rep[ k ][ i ] ]:
            if fake[ lf ]:
                continue
            pts = self._V[ ids[ off[ lf ] : off[ lf + 1 ] ] ]
            if len( pts ) >= 3:
                b = _bases( pts, np.array( [ 0, len( pts ) ] ), np.arange( len( pts ) ), 2 )[ 0 ]
                q = pts - pts.mean( axis = 0 )
                res.append( pts[ np.argsort( np.arctan2( q @ b[ 1 ], q @ b[ 0 ] ) ) ] )
        return res

    # ---- the numbers: tensors of the framework, differentiable ( `CellGeometry` ) -------------------

    def geometry( self, positions = None, weights = None ) -> "CellGeometry":
        """The numbers of this complex for given seeds ( `CellGeometry` ): measures, centroids, normals, seed vectors, vertex positions,
        simplices, distances -- tensors of the framework of `positions`, differentiable with respect to them and to `weights`. By
        default the positions and weights the complex was built from ( what `measures`, `centroids`... read ); others keep THIS
        topology -- the way to differentiate, or to evaluate under a `jax.jit`, with the complex built outside."""
        from .CellGeometry import CellGeometry
        if positions is None:
            if "default" not in self._geom:
                # ( cells without seeds, `CellComplex( cells )`: their vertices as they were cut, numpy constants )
                self._geom[ "default" ] = CellGeometry( self, *( self._values or ( None, None ) ) )
            return self._geom[ "default" ]
        return CellGeometry( self, positions, weights )

    def simplices( self, k = None, ids = None ):
        """A decomposition of the `k`-faces ( the cells by default; `ids`: those faces only ) into `k`-simplices: `( points [ m,
        k + 1, d ], faces [ m ] )`, the corners of each simplex and the face it belongs to. Each face is fanned from the mean of its
        vertices over its faces one dimension below, themselves fanned from theirs, down to the edges: a simplex per chain
        `face > ... > edge`. Exact: the volumes of the simplices of a face add up to its measure. Unbounded faces have none; with a
        periodicity, the simplices of a face are those of the copy its geometry is read off ( `Face.points` ). `points`: a tensor of
        the framework of the positions, differentiable."""
        pts, owner = self.geometry().simplices( k )
        if ids is None:
            return pts, owner
        sel = np.flatnonzero( np.isin( owner, np.asarray( ids, dtype = np.int64 ).reshape( -1 ) ) )
        return pts[ sel ], owner[ sel ]

    def locate( self, points ):
        """`( cells [ m ], moved [ m, d ] )`: the cell that holds each point ( `-1` outside the domain ), and the point moved by the
        group to the copy of that cell ( `Face.points` ) -- the point itself without a periodicity, a tensor of its framework. Exact:
        the cell of the smallest power distance, found by a nearest-neighbour search among the seeds lifted in one more dimension
        ( `|x - p|^2 - w = |( x, 0 ) - ( p, sqrt( W - w ) )|^2 - W` ), and among their images with a periodicity."""
        from .CellGeometry import _affine
        from loom.tensor.host import to_host
        raw = getattr( points, "raw", points )
        cells, _, maps = self._locate( np.asarray( to_host( raw ), dtype = float ).reshape( -1, self._d ) )
        moved = _affine( raw, maps )
        return cells, moved

    def boundary_distances( self, points, cells = None ):
        """`( distances [ m ], facets [ m ] )`: for each point, its distance to the boundary of its cell -- positive inside, the
        distance to the plane of the nearest facet -- and that facet. `cells`: the cell of each point, the points then being in the
        frame of the copies of these cells ( `locate`'s `moved` ); by default, `locate` finds them. A point outside the domain has
        NaN and `-1`; a point given outside its cell, minus the largest overshoot of a plane ( a lower bound of its distance ).
        `distances`: a tensor of the framework of the points, differentiable with respect to them and to the seeds."""
        return self.geometry().boundary_distances( points, cells )

    def _locate( self, P ):
        """`( cells, moved, maps )` on the host: the cells of the points `P`, the points moved to the copies of their cells, and the
        affine maps that did it ( `[ m, d, d + 1 ]`, NaN outside the domain )"""
        if self._seeds is None or not self._power:
            raise ValueError( "locate: this complex does not know the power cells of its seeds ( no `seeds`, or groups )" )
        from scipy.spatial import cKDTree
        from .periodicity import _compose as compose_one
        d, n = self._d, self._n
        S, w = self._seeds, self._weights
        eye = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
        G = self._group.maps if self._group is not None else eye[ None ]
        if self._group is not None:
            Y, M, c = self._group.fold( P )
            Y = np.asarray( Y, dtype = float )
            M = np.broadcast_to( np.eye( d ), ( len( P ), d, d ) ) if M is None else np.asarray( M, dtype = float ).reshape( -1, d, d )
            F = np.concatenate( [ M, np.asarray( c, dtype = float ).reshape( -1, d, 1 ) ], axis = 2 )
        else:
            Y, F = P, np.broadcast_to( eye, ( len( P ), d, d + 1 ) )
        cand = np.concatenate( [ S @ A[ :, :d ].T + A[ :, d ] for A in G ] )
        lift = np.sqrt( np.maximum( w.max( initial = 0 ) - w, 0 ) )
        tree = cKDTree( np.concatenate( [ cand, np.tile( lift, len( G ) )[ :, None ] ], axis = 1 ) )
        _, idx = tree.query( np.concatenate( [ Y, np.zeros( ( len( Y ), 1 ) ) ], axis = 1 ) )
        cell, t = idx % n, idx // n
        # the point in the frame of the cut of that cell ( the inverse of its image ), then in the frame of its copy
        maps = _compose_maps( _inverse_maps( G[ t ] ), F )
        if self._maps is not None:
            maps = _compose_maps( self._maps[ cell ], maps )
        moved = np.einsum( "mij,mj->mi", maps[ :, :, :d ], P ) + maps[ :, :, d ]
        Z = np.einsum( "mji,mj->mi", G[ t ][ :, :, :d ], Y - G[ t ][ :, :, d ] )
        inside = np.ones( len( P ), bool )
        if self.walls is not None and len( self.walls[ 1 ] ):
            dirs, offs = self.walls
            nrm = np.linalg.norm( dirs, axis = 1 )
            inside = ( ( Z @ dirs.T - offs ) / nrm <= self.tolerance ).all( axis = 1 )
        cell = np.where( inside, cell, -1 )
        moved[ ~inside ] = np.nan
        maps[ ~inside ] = np.nan
        return cell, moved, maps

    def _nearest_local_facets( self, P, cells ):
        """`( distances, facets, local facets )` on the host, for points `P` in the frame of the copies of their `cells`: the nearest
        plane of the facets of each cell ( deduplicated local facets, `_tables` ), and the global facet it is"""
        T = self._tables()
        nrm, off, du_cell, csr, glob = T[ "facet_planes" ]
        m = len( P )
        res, fac, loc = np.full( m, np.nan ), np.full( m, -1, np.int64 ), np.full( m, -1, np.int64 )
        ok = cells >= 0
        ok[ ok ] &= du_cell[ cells[ ok ] ] >= 0
        c = np.where( ok, du_cell[ np.maximum( cells, 0 ) ], 0 )
        o, lf = csr
        cnt = np.where( ok, np.diff( o )[ c ], 0 )
        pt = np.repeat( np.arange( m ), cnt )
        f = lf[ np.repeat( o[ c ], cnt ) + np.arange( cnt.sum() ) - np.repeat( np.cumsum( cnt ) - cnt, cnt ) ]
        dist = off[ f ] - ( nrm[ f ] * P[ pt ] ).sum( axis = 1 )
        order = np.lexsort( ( dist, pt ) )
        first = order[ np.r_[ True, pt[ order ][ 1: ] != pt[ order ][ :-1 ] ] ] if len( order ) else order
        res[ pt[ first ] ] = dist[ first ]
        fac[ pt[ first ] ] = glob[ f[ first ] ]
        loc[ pt[ first ] ] = f[ first ]
        return res, fac, loc

    def _tables( self ):
        """The host tables of `CellGeometry` ( see its module ): the rows, the deduplicated local faces ( one per cell and global face,
        their cleaned vertices ), the apexes, the fans of the representatives and of the local facets, the seed vectors"""
        if "tables" in self._geom:
            return self._geom[ "tables" ]
        d, n = self._d, self._n
        V, gv, vcell = self._V, self._gv, self._vcell
        VC, fake_v = self._ctx[ "VC" ], self._ctx[ "fake_v" ]
        eye = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
        T = {}

        # THE ROWS: the local vertices that are corners
        rows = np.flatnonzero( ( gv >= 0 ) & ~fake_v )
        row_of = np.full( len( V ), -1, np.int64 )
        row_of[ rows ] = np.arange( len( rows ) )
        T[ "own" ] = vcell[ rows ].astype( np.int32 )
        T[ "frame" ] = self._maps[ vcell[ rows ] ] if self._maps is not None else np.broadcast_to( eye, ( len( rows ), d, d + 1 ) ).copy()
        T.update( self._vertex_equations( rows ) )
        Vr = V[ rows ]
        vr = np.full( self.nb_faces( 0 ), -1, np.int64 )
        gr = gv[ rows ]
        vr[ gr[ ::-1 ] ] = np.arange( len( rows ) )[ ::-1 ]
        T[ "vertex_rows" ] = vr.astype( np.int32 )
        T[ "row_of" ] = row_of
        T[ "rows" ] = rows

        # THE LOCAL FACES, codimension by codimension: one per cell and global face ( the true lattice of the cell, `Glue.h` ), their
        # rows ( the names `du` / `nbdu` are kept from when they had to be deduplicated )
        du, rcsr, nbdu = {}, {}, {}
        for c in range( d ):
            L = self._loc[ c ]
            nl = len( L[ "keys" ] )
            r = row_of[ L[ "lv" ] ]
            okp = r >= 0                                 # ( `lf` is sorted: so are the pairs kept )
            f = L[ "lf" ][ okp ]
            off = np.concatenate( [ [ 0 ], np.cumsum( np.bincount( f, minlength = nl ) ) ] )
            valid = ( np.diff( off ) > 0 ) & ( L[ "lg" ] >= 0 )
            du[ c ] = np.where( valid, np.arange( nl ), -1 )
            nbdu[ c ] = nl
            rcsr[ c ] = ( off, r[ okp ] )
        kcsr = {}
        for c in range( d - 1 ):
            par, chi = self._lchild[ c ]
            a, b = du[ c ][ par ], du[ c + 1 ][ chi ]
            ok = ( a >= 0 ) & ( b >= 0 )
            key = np.unique( a[ ok ] * ( nbdu[ c + 1 ] + 1 ) + b[ ok ] )   # ( a child found from two facets: once )
            kcsr[ c ] = _csr( np.stack( [ key // ( nbdu[ c + 1 ] + 1 ), key % ( nbdu[ c + 1 ] + 1 ) ], axis = 1 ), nbdu[ c ] )
        # the apexes: the dedup faces of dimension >= 2, their rows
        base, offs, rws, hp = {}, [ 0 ], [], []
        for c in range( d - 1 ):
            base[ c ] = offs[ -1 ] - 0 if c == 0 else base[ c - 1 ] + nbdu[ c - 1 ]
            o, r = rcsr[ c ]
            offs = offs + list( offs[ -1 ] + o[ 1: ] )
            rws.append( r )
            own = np.repeat( np.arange( nbdu[ c ] ), np.diff( o ) )
            mean = np.zeros( ( nbdu[ c ], d ) )
            np.add.at( mean, own, Vr[ r ] )
            hp.append( mean / np.maximum( np.diff( o ), 1 )[ :, None ] )
        T[ "apex_off" ] = np.asarray( offs, np.int64 )
        T[ "apex_rows" ] = np.concatenate( rws ) if rws else np.zeros( 0, np.int64 )
        apex_host = np.concatenate( hp ) if hp else np.zeros( ( 0, d ) )

        def chains( c0, start ):
            """the fans of dedup faces of codimension `c0`: `( corners [ ns, d - c0 + 1 ], owner [ ns ] )`"""
            owner, cur, cols, c = np.arange( len( start ) ), np.asarray( start, np.int64 ), [], c0
            while d - c >= 2:
                cols.append( -1 - ( base[ c ] + cur ) )
                o, kids = kcsr[ c ]
                cnt = np.diff( o )[ cur ]
                at = np.repeat( o[ cur ], cnt ) + np.arange( cnt.sum() ) - np.repeat( np.cumsum( cnt ) - cnt, cnt )
                owner, cur = np.repeat( owner, cnt ), kids[ at ]
                cols = [ np.repeat( x, cnt ) for x in cols ]
                c += 1
            o, r = rcsr[ d - 1 ]
            ok = np.diff( o )[ cur ] == 2
            owner, cur, cols = owner[ ok ], cur[ ok ], [ x[ ok ] for x in cols ]
            cols += [ r[ o[ cur ] ], r[ o[ cur ] + 1 ] ]
            return np.stack( cols, axis = 1 ), owner

        def host_points( corners ):
            return np.where( ( corners >= 0 )[ :, :, None ], Vr[ np.maximum( corners, 0 ) ], apex_host[ np.maximum( -1 - corners, 0 ) ] )

        def area_vectors( P ):
            return _area_vectors( P[ :, 1: ] - P[ :, :1 ] )

        # the fans of the representatives of the bounded faces
        normals = self._sides_and_normals()[ 1 ] if d >= 2 else None
        T[ "simplices" ] = {}
        for k in range( 1, d + 1 ):
            c0 = d - k
            ids = np.flatnonzero( ~self._unbounded[ k ] & ( self._rep[ k ] >= 0 ) )
            start = du[ c0 ][ self._rep[ k ][ ids ] ] if c0 < d else ids
            # ( an empty cell, a face without a corner: no fan )
            has = start >= 0
            ids, start = ids[ has ], start[ has ]
            corners, owner = chains( c0, start )
            P = host_points( corners )
            signs = np.ones( len( corners ) )
            if k == d - 1 and d >= 2 and len( corners ):
                signs = np.where( ( area_vectors( P ) * normals[ ids[ owner ] ] ).sum( axis = 1 ) < 0, -1.0, 1.0 )
            E = P[ :, 1: ] - P[ :, :1 ]
            vol = np.sqrt( np.maximum( np.linalg.det( np.einsum( "mid,mjd->mij", E, E ) ), 0 ) ) / math.factorial( k ) if len( E ) else np.zeros( 0 )
            meas = np.bincount( owner, weights = vol, minlength = len( ids ) )
            T[ "simplices" ][ k ] = dict( corners = corners, signs = signs, owner = ids[ owner ], faces = ids,
                                          s0 = np.searchsorted( owner, np.arange( len( ids ) ) ).astype( np.int32 ),
                                          s1 = np.searchsorted( owner, np.arange( len( ids ) ), side = "right" ).astype( np.int32 ),
                                          keep = vol > 1e-12 * meas[ owner ] if len( vol ) else np.zeros( 0, bool ) )

        # the local facets ( dedup ), outward of their cell: their planes on the host, and their fans for `CellGeometry`
        if d >= 2:
            L1 = self._loc[ 1 ]
            o1, r1 = rcsr[ 1 ] if d > 1 else ( None, None )
            cell_of = np.zeros( nbdu[ 1 ], np.int64 )
            glob = np.zeros( nbdu[ 1 ], np.int64 )
            ok = du[ 1 ] >= 0
            cell_of[ du[ 1 ][ ok ] ] = L1[ "cell" ][ ok ]
            glob[ du[ 1 ][ ok ] ] = L1[ "lg" ][ ok ]
            inside = np.zeros( ( n, d ) )
            np.add.at( inside, vcell, V )
            inside /= np.maximum( np.bincount( vcell, minlength = n ), 1 )[ :, None ]
            cut_of = np.zeros( nbdu[ 1 ], np.int64 )
            cut_of[ du[ 1 ][ ok ] ] = L1[ "keys" ][ ok, 0 ]
            nrm = self._cut_planes()[ cut_of, :d ].copy()     # ( exact, outward of the cell; then in the frame of the user )
            if self._maps is not None and nbdu[ 1 ]:
                nrm = np.einsum( "fij,fj->fi", self._maps[ cell_of ][ :, :, :d ], nrm )
            ctr = np.zeros( ( nbdu[ 1 ], d ) )               # ( a point of each facet; none for a facet without finite row )
            has = np.diff( o1 ) > 0
            ctr[ has ] = Vr[ r1[ o1[ :-1 ][ has ] ] ]
            du_cell = np.full( n, -1, np.int64 )             # ( the dedup cell of each seed; an empty cell has none )
            okc = du[ 0 ] >= 0
            du_cell[ self._loc[ 0 ][ "cell" ][ okc ] ] = du[ 0 ][ okc ]
            par, chi = self._lchild[ 0 ]
            a, b = du[ 0 ][ par ], du[ 1 ][ chi ]
            okk = ( a >= 0 ) & ( b >= 0 )
            T[ "facet_planes" ] = ( nrm, ( nrm * ctr ).sum( axis = 1 ), du_cell, _csr( _unique_pairs( a[ okk ], b[ okk ] ), nbdu[ 0 ] ), glob )

            def local_facets( lf ):
                corners, owner = chains( 1, lf )
                P = host_points( corners )
                signs = np.where( ( area_vectors( P ) * nrm[ np.asarray( lf )[ owner ] ] ).sum( axis = 1 ) < 0, -1.0, 1.0 ) \
                    if d >= 2 and len( corners ) else np.ones( len( corners ) )
                return corners, signs, owner
            T[ "local_facets" ] = local_facets

            # the seed vectors: from the representative's cell to the image across
            rep = self._rep[ d - 1 ]
            cut = L1[ "keys" ][ rep, 0 ]
            a = L1[ "cell" ][ rep ]
            self._sides_and_normals()
            fr = self._maps[ a ][ :, :, :d ] if self._maps is not None else np.broadcast_to( np.eye( d ), ( len( rep ), d, d ) )
            j = self._gen[ cut ]
            T[ "seed_vectors" ] = dict( a = a.astype( np.int32 ), j = np.maximum( j, 0 ).astype( np.int32 ),
                                        t = self._img[ cut ].astype( np.int32 ), wall = j < 0,
                                        sign = np.where( self._geom[ "swap" ], -1.0, 1.0 ), frame = np.ascontiguousarray( fr ) )
        self._geom[ "tables" ] = T
        return T

    def _vertex_equations( self, rows ):
        """THE EQUATIONS OF EACH ROW ( `sdot/complex/Geometry.h::vertex_system` ): the bisectors of two images of seeds, or walls, in the
        frame of the row's cell -- `eq_a`, `eq_ea` ( seed, element ), `eq_b`, `eq_eb` ( `eq_b < 0`: the wall `-1 - eq_b` ), with the
        tables `maps` ( the elements, the group's first ) and `walls`. By default the `d` cuts of the row's own vertex. Where they
        are ill-conditioned -- a degenerate vertex, where the cell kept `d` of the planes through it that are not independent ( the
        corner of a cube and a diagonal bisector ) -- the `d` best conditioned among the generators of ALL the copies of the vertex,
        brought into the frame of the row ( `x_row = E_row E_o^-1 x_o` )."""
        d, n = self._d, self._n
        VC, vcell, gv = self._ctx[ "VC" ], self._vcell, self._gv
        eye = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
        G = self._group.maps if self._group is not None else eye[ None ]
        elems, ekeys = [ g for g in G ], { tuple( _round_key( g.reshape( -1 ) ) ): i for i, g in enumerate( G ) }
        def elem( M ):
            k = tuple( _round_key( M.reshape( -1 ) ) )
            if k not in ekeys:
                ekeys[ k ] = len( elems )
                elems.append( np.asarray( M, dtype = float ) )
            return ekeys[ k ]
        walls = [ np.concatenate( [ self.walls[ 0 ], self.walls[ 1 ][ :, None ] ], axis = 1 ) ] if self.walls is not None and len( self.walls[ 1 ] ) \
            else [ np.zeros( ( 1, d + 1 ) ) ]
        nb_walls = [ len( walls[ 0 ] ) ]
        wkeys = {}
        def wall( plane ):
            k = tuple( _round_key( plane ) )
            if k not in wkeys:
                wkeys[ k ] = nb_walls[ 0 ]
                walls.append( np.asarray( plane, dtype = float )[ None ] )
                nb_walls[ 0 ] += 1
            return wkeys[ k ]

        m = len( rows )
        own = vcell[ rows ]
        eq_a = np.repeat( own[ :, None ], d, axis = 1 )
        eq_ea = np.zeros( ( m, d ), np.int64 )
        cuts = np.where( self._veq[ rows ] >= 0, self._veq[ rows ], VC[ rows ] ) if hasattr( self, "_veq" ) else VC[ rows ]
        eq_b = self._gen[ cuts ].reshape( m, d )
        eq_eb = np.where( eq_b >= 0, self._img[ cuts ].reshape( m, d ), 0 )

        # the conditioning of the default equations: the determinant of their unit normals ( host seeds, the frame of the cuts )
        if self._seeds is None:
            return dict( eq_a = eq_a, eq_ea = eq_ea, eq_b = eq_b, eq_eb = eq_eb, maps = G, walls = walls[ 0 ] )
        S = self._seeds
        W0 = walls[ 0 ]
        def normals( a, ea, b, eb ):
            Ea, Eb = np.stack( elems )[ ea ], np.stack( elems )[ eb ]
            qa = np.einsum( "...ij,...j->...i", Ea[ ..., :d ], S[ a ] ) + Ea[ ..., d ]
            qb = np.einsum( "...ij,...j->...i", Eb[ ..., :d ], S[ np.maximum( b, 0 ) ] ) + Eb[ ..., d ]
            Wt = np.concatenate( walls )
            nrm = np.where( ( b >= 0 )[ ..., None ], qb - qa, Wt[ np.maximum( -1 - b, 0 ) ][ ..., :d ] )
            return nrm / np.maximum( np.linalg.norm( nrm, axis = -1, keepdims = True ), 1e-300 )
        det = np.abs( np.linalg.det( normals( eq_a, eq_ea, eq_b, eq_eb ) ) ) if m else np.zeros( 0 )
        bad = np.flatnonzero( det < 1e-6 )
        if len( bad ):
            occ = {}
            real = np.flatnonzero( gv >= 0 )
            for o in real:
                occ.setdefault( int( gv[ o ] ), [] ).append( o )
            E = self._E
            for r in bad:
                o1 = rows[ r ]
                cand = []                                 # ( a, ea, b, eb ) in the frame of o1
                for o in occ[ int( gv[ o1 ] ) ]:
                    H = _compose_maps( E[ o1 ], _inverse_maps( E[ o ] ) )
                    s_o = int( vcell[ o ] )
                    for q in range( d ):
                        cut = VC[ o, q ]
                        j = int( self._gen[ cut ] )
                        if j >= 0:
                            cand.append( ( s_o, elem( H ), j, elem( _compose_maps( H, G[ int( self._img[ cut ] ) ] ) ) ) )
                        else:
                            Hi = _inverse_maps( H )                # ( the wall `n . x_o <= o_w`, `x_o = H^-1 x` )
                            nw, ow = W0[ -1 - j, :d ], W0[ -1 - j, d ]
                            cand.append( ( 0, 0, -1 - wall( np.concatenate( [ Hi[ :, :d ].T @ nw, [ ow - nw @ Hi[ :, d ] ] ] ) ), 0 ) )
                cand = list( dict.fromkeys( cand ) )
                C = np.asarray( cand, np.int64 )
                N = normals( C[ :, 0 ], C[ :, 1 ], C[ :, 2 ], C[ :, 3 ] )
                chosen = []                               # ( greedy: the largest residual after the ones chosen )
                for _ in range( d ):
                    res = N.copy()
                    if chosen:
                        Q, _ = np.linalg.qr( N[ chosen ].T )
                        res = N - ( N @ Q ) @ Q.T
                    chosen.append( int( np.argmax( np.linalg.norm( res, axis = 1 ) ) ) )
                eq_a[ r ], eq_ea[ r ], eq_b[ r ], eq_eb[ r ] = C[ chosen ].T
        return dict( eq_a = eq_a, eq_ea = eq_ea, eq_b = eq_b, eq_eb = eq_eb, maps = np.stack( elems ), walls = np.concatenate( walls ) )

    def _local_planes( self ):

        """`( normals, offsets, valid )` of the local facets, outward of their cell, in the frame of the user: the facets that are
        faces of the complex ( not flat, not on a made-up wall )"""
        if "planes" in self._geom:
            return self._geom[ "planes" ]
        d, V, L = self._d, self._V, self._loc[ 1 ]
        off, ids = self._local_csr( 1 )
        m = len( L[ "keys" ] )
        inside = np.zeros( ( self._n, d ) )
        np.add.at( inside, self._vcell, V )
        inside /= np.maximum( np.bincount( self._vcell, minlength = self._n ), 1 )[ :, None ]
        ctr = V[ ids[ np.minimum( off[ :-1 ], max( len( ids ) - 1, 0 ) ) ] ] if m else np.zeros( ( 0, d ) )
        if m == 0:
            nrm = np.zeros( ( 0, d ) )
        elif d == 1:
            nrm = np.sign( ctr - inside[ L[ "cell" ] ] )
        else:
            nrm = _bases( V, off, ids, d )[ :, d - 1 ]
        flip = ( ( inside[ L[ "cell" ] ] - ctr ) * nrm ).sum( axis = 1 ) > 0
        nrm[ flip ] *= -1
        res = ( nrm, ( nrm * ctr ).sum( axis = 1 ), ( L[ "lg" ] >= 0 ) & ~L[ "fake" ] )
        self._geom[ "planes" ] = res
        return res

    # ---- geometry ----------------------------------------------------------------------------------

    def _measures_and_centroids( self, k ):
        """`( measures, centroids )` of the `k`-faces: those of their representatives"""
        if k == 0:
            return np.ones( self.nb_faces( 0 ) ), self._vertex_positions
        meas, cen = self._local_geom( self._d - k )
        rep = self._rep[ k ]
        has = rep >= 0                                   # ( an empty cell: no copy, nothing to measure )
        r = np.maximum( rep, 0 )
        return np.where( has, meas[ r ] if len( meas ) else 0.0, 0.0 ), np.where( has[ :, None ], cen[ r ] if len( cen ) else np.nan, np.nan )

    def _sides_and_normals( self ):
        """`( sides [ nf, 2 ], normals [ nf, d ] )`: the seeds on both sides of each facet ( the smaller first; `-1 - wall` for
        a wall ), and the unit normal from the first to the second -- the plane of the cut of its representative, in the frame of
        the user"""
        if "sides" in self._geom:
            return self._geom[ "sides" ]
        d = self._d
        L = self._loc[ 1 ]
        rep = self._rep[ d - 1 ]
        a, b = L[ "cell" ][ rep ], self._gen[ L[ "keys" ][ rep, 0 ] ]
        swap = ( b >= 0 ) & ( b < a )
        sides = np.stack( [ np.where( swap, b, a ), np.where( swap, a, b ) ], axis = 1 )

        # the plane of its cut, exact and outward of its cell ( `_cut_planes` ), in the frame of the user
        nrm = self._cut_planes()[ L[ "keys" ][ rep, 0 ], :d ].copy() if len( rep ) else np.zeros( ( 0, d ) )
        if self._maps is not None and len( rep ):
            nrm = np.einsum( "fij,fj->fi", self._maps[ a ][ :, :, :d ], nrm )
        nrm[ swap ] *= -1
        self._geom[ "sides" ] = ( sides, nrm )
        self._geom[ "swap" ] = swap
        return sides, nrm


_NAMES = { ( 1, 0 ): "vertices", ( 1, 1 ): "cells", ( 2, 0 ): "vertices", ( 2, 1 ): "edges", ( 2, 2 ): "cells",
           ( 3, 0 ): "vertices", ( 3, 1 ): "edges", ( 3, 2 ): "facets", ( 3, 3 ): "cells" }


class Faces( Displayable ):
    """The `k`-faces of a `CellComplex`: a sequence of `Face` ( `len`, `[ i ]`, iteration ), and their values as arrays.

    For the facets: `sides` and `normals`. For the cells: `seeds` ( their index, which is the seed's )."""

    def __init__( self, complex, k ):
        self.complex = complex
        self.dim = k

    def __len__( self ):
        return self.complex.nb_faces( self.dim )

    def __getitem__( self, i ):
        if isinstance( i, slice ):
            return [ Face( self.complex, self.dim, j ) for j in range( *i.indices( len( self ) ) ) ]
        i = int( i )
        if i < 0:
            i += len( self )
        if not 0 <= i < len( self ):
            raise IndexError( f"{ self.dim }-face { i }: there are { len( self ) }" )
        return Face( self.complex, self.dim, i )

    def __iter__( self ):
        return ( Face( self.complex, self.dim, i ) for i in range( len( self ) ) )

    def __repr__( self ):
        return f"Faces( dim = { self.dim }, { len( self ) } faces )"

    def incident( self, j ):
        """The `j`-faces of each of these faces, as a CSR `( offsets, ids )` ( see `CellComplex.incident` )."""
        return self.complex.incident( self.dim, j )

    def simplices( self, ids = None ):
        """`( points [ m, dim + 1, d ], faces [ m ] )`: these faces ( or those of `ids` ) cut into simplices ( see
        `CellComplex.simplices` )."""
        return self.complex.simplices( self.dim, ids )

    def children( self ):
        """`incident( dim - 1 )`: the faces of each one, one dimension below."""
        return self.incident( self.dim - 1 )

    def parents( self ):
        """`incident( dim + 1 )`: the faces that contain each one, one dimension above."""
        return self.incident( self.dim + 1 )

    @property
    def measures( self ):
        """`[ m ]`: length, area, volume... ( `1` for a vertex; infinite for an unbounded face ). Like all the numbers of the complex, a
        tensor of the framework of the positions, differentiable with respect to the positions and the weights ( `CellComplex.geometry` )."""
        return self.complex.geometry().measures( self.dim )

    @property
    def centroids( self ):
        """`[ m, d ]`: the centers of mass ( NaN for an unbounded face )."""
        return self.complex.geometry().centroids( self.dim )

    @property
    def is_bounded( self ):
        """`[ m ]` booleans: whether each face is bounded ( a diagram without a domain has unbounded cells )."""
        return ~self.complex._unbounded[ self.dim ]

    @property
    def sides( self ):
        """For the facets: `[ m, 2 ]`, the seeds on both sides, the smaller first; the second is `-1 - wall` on the
        boundary of the domain ( `CellComplex.walls` )."""
        self._need_facets( "sides" )
        return self.complex._sides_and_normals()[ 0 ]

    @property
    def normals( self ):
        """For the facets: `[ m, d ]`, the unit normal, from the first side to the second ( outward on the boundary )."""
        self._need_facets( "normals" )
        return self.complex.geometry().normals

    @property
    def seed_vectors( self ):
        """For the facets: `[ m, d ]`, from the seed `sides[ 0 ]` to the seed `sides[ 1 ]` -- with a periodicity, to the image of
        `sides[ 1 ]` across the facet ( a cell can meet another, or itself, across the period ). Zero on the boundary of the domain.
        What a finite-volume scheme divides the area of a facet by: `measures / norm( seed_vectors )`."""
        self._need_facets( "seed_vectors" )
        return self.complex.geometry().seed_vectors

    @property
    def seeds( self ):
        """For the cells: `[ n ]`, the seed of each ( its index )."""
        if self.dim != self.complex.dim:
            raise AttributeError( "`seeds`: the cells only" )
        return np.arange( len( self ) )

    def _need_facets( self, what ):
        if self.dim != self.complex.dim - 1:
            raise AttributeError( f"`{ what }`: the facets only ( dimension { self.complex.dim - 1 } ), not { self.dim }-faces" )

    def add_to_viz( self, viz, ids = None, color = None, opacity = 1.0, faces = True, edges = True, edge_color = None, frames = True ):
        """Draws these faces ( or those of `ids` ), the bounded ones: the vertices as points, the edges as segments, the
        `2`-faces as polygons, the `3`-faces by their `2`-faces. `color`: one for all, by default one per face ( on a block
        reserved for all the faces of this dimension: the cell of a seed always has the same ). `faces = False`: the edges of
        the polygons only ( a wireframe ), in `edge_color` ( by default a darker `color` )."""
        cx, k = self.complex, self.dim
        ids = np.arange( len( self ) ) if ids is None else np.asarray( ids, dtype = np.int64 ).reshape( -1 )
        ids = ids[ self.is_bounded[ ids ] ]
        first = viz.reserve_colors( len( self ) ) if color is None else None
        col = lambda i: color if color is not None else viz.color_at( first + int( i ) )
        if k == 0:
            for i in ids:
                viz.add_points( cx._vertex_positions[ [ i ] ], color = col( i ), opacity = opacity, frames = frames )
            return viz
        for i in ids:
            if k == 1:
                viz.add_edges( Face( cx, 1, int( i ) )._host_points(), color = col( i ), opacity = opacity, frames = frames )
                continue
            for pts in cx._local_polygons( k, int( i ) ) if k <= 3 else []:
                if len( pts ) < 3:
                    continue
                if faces:
                    viz.add_faces( pts, [ list( range( len( pts ) ) ) ], color = col( i ), opacity = opacity, frames = frames )
                if edges or not faces:
                    viz.add_edges( pts, color = edge_color or viz.darker( col( i ) ), opacity = 1.0 if not faces else opacity,
                                   closed = True, frames = frames )
        return viz


class Face( Displayable ):
    """ONE face of a `CellComplex`: its dimension `dim`, its number `index` among the faces of that dimension, and what it
    is attached to -- `children`, `parents`, `faces( j )` -- as `Face`s again, to walk the complex."""

    def __init__( self, complex, k, i ):
        self.complex = complex
        self.dim = k
        self.index = i

    def __repr__( self ):
        return f"Face( dim = { self.dim }, index = { self.index } )"

    def __eq__( self, other ):
        return isinstance( other, Face ) and other.complex is self.complex and ( other.dim, other.index ) == ( self.dim, self.index )

    def __hash__( self ):
        return hash( ( id( self.complex ), self.dim, self.index ) )

    def ids( self, j ):
        """The numbers of the `j`-faces of this one ( contained for `j < dim`, containing for `j > dim` ), an int array."""
        off, ids = self.complex.incident( self.dim, j )
        return ids[ off[ self.index ] : off[ self.index + 1 ] ]

    def faces( self, j ):
        """The `j`-faces of this one, as `Face`s ( contained for `j < dim`, containing for `j > dim` )."""
        return [ Face( self.complex, j, int( i ) ) for i in self.ids( j ) ]

    @property
    def children( self ):
        """Its `dim - 1`-faces ( none for a vertex )."""
        return self.faces( self.dim - 1 ) if self.dim > 0 else []

    @property
    def parents( self ):
        """The `dim + 1`-faces that contain it ( none for a cell )."""
        return self.faces( self.dim + 1 ) if self.dim < self.complex.dim else []

    @property
    def vertices( self ):
        """Its vertices, as `Face`s -- in the order of its boundary for a `2`-face."""
        return self.faces( 0 ) if self.dim else [ self ]

    @property
    def cells( self ):
        """The cells that contain it, as `Face`s ( their `index` is the seed )."""
        return self.faces( self.complex.dim ) if self.dim < self.complex.dim else [ self ]

    @property
    def points( self ):
        """`[ nb_vertices, d ]`: the positions of its vertices, in the order of `vertices` -- read off ONE copy of the face ( with a
        periodicity: the copy of one of its cells, next to the given position of its seed )."""
        cx = self.complex
        rows = cx._tables()[ "vertex_rows" ][ [ self.index ] ] if self.dim == 0 else cx._tables()[ "row_of" ][ self._host_vertex_ids() ]
        return cx.geometry()._points( np.asarray( rows )[ :, None ] )[ :, 0 ]

    def _host_vertex_ids( self ):
        off, lv = self.complex._rep_points( self.dim )
        return lv[ off[ self.index ] : off[ self.index + 1 ] ]

    def _host_points( self ):
        """`points` on the host ( to draw )"""
        if self.dim == 0:
            return self.complex._vertex_positions[ [ self.index ] ]
        return self.complex._V[ self._host_vertex_ids() ]

    @property
    def measure( self ):
        """Its length, area, volume... ( `1` for a vertex, infinite when unbounded ): a scalar of the framework, differentiable."""
        return Faces( self.complex, self.dim ).measures[ self.index ]

    @property
    def centroid( self ):
        """Its center of mass, `[ d ]`."""
        return Faces( self.complex, self.dim ).centroids[ self.index ]

    @property
    def is_bounded( self ):
        return not bool( self.complex._unbounded[ self.dim ][ self.index ] )

    @property
    def sides( self ):
        """For a facet: `( a, b )`, the seeds on both sides, `b = -1 - wall` on the boundary of the domain."""
        return tuple( int( s ) for s in Faces( self.complex, self.dim ).sides[ self.index ] )

    @property
    def normal( self ):
        """For a facet: its unit normal, from `sides[ 0 ]` to `sides[ 1 ]`."""
        return Faces( self.complex, self.dim ).normals[ self.index ]

    @property
    def seed_vector( self ):
        """For a facet: from the seed `sides[ 0 ]` to ( the image of ) the seed `sides[ 1 ]` across it, zero on a wall."""
        return Faces( self.complex, self.dim ).seed_vectors[ self.index ]

    @property
    def seed( self ):
        """For a cell: its seed."""
        if self.dim != self.complex.dim:
            raise AttributeError( "`seed`: a cell only" )
        return self.index

    def simplices( self ):
        """`[ m, dim + 1, d ]`: a decomposition of this face into simplices ( see `CellComplex.simplices` )."""
        return self.complex.simplices( self.dim, [ self.index ] )[ 0 ]

    def add_to_viz( self, viz, **kwargs ):
        """Draws this face ( see `Faces.add_to_viz` )."""
        return Faces( self.complex, self.dim ).add_to_viz( viz, ids = [ self.index ], **kwargs )


# ---- the gluing ------------------------------------------------------------------------------------

def _bulk( cells ):
    """The cells of a batch, flattened in ONE read: `( V [ nv, d ], vcell [ nv ], VC [ nv, d ], CID [ nc ], ccell [ nc ], coff )` --
    the vertices of all the items, the item of each, the `d` cuts of each ( indices into the flat cuts ), the identifiers of the
    cuts, the item of each, and the first cut of each item."""
    from loom.tensor.host import to_host
    d, n, nbb = int( cells.dim ), int( cells.nb_items ), len( cells.batch_axes or [] )

    def rows( t ):
        raw = np.asarray( to_host( t.raw ) )
        return raw.reshape( ( -1, ) + raw.shape[ nbb : ] )[ :n ]

    def counts( sv ):
        v = np.atleast_1d( np.asarray( sv.value ) ).reshape( -1 ).astype( np.int64 )
        return v[ :n ] if v.size >= n and v.size > 1 else np.full( n, int( v[ 0 ] ) )

    if cells.vertex_positions.raw is None:
        e = np.zeros( 0, np.int64 )
        return np.zeros( ( 0, d ) ), e, np.zeros( ( 0, d ), np.int64 ), e, e, np.zeros( n + 1, np.int64 )
    vp, cid = rows( cells.vertex_positions ), rows( cells.cut_ids )
    nv, nc = counts( cells.nb_vertices ), counts( cells.nb_cuts )
    vmask = np.arange( vp.shape[ 1 ] )[ None, : ] < nv[ :, None ]
    cmask = np.arange( cid.shape[ 1 ] )[ None, : ] < nc[ :, None ]
    coff = np.concatenate( [ [ 0 ], np.cumsum( nc ) ] )
    V = vp[ vmask ].astype( float ).reshape( -1, d )
    vcell = np.repeat( np.arange( n ), nv )
    CID = cid[ cmask ].astype( np.int64 )
    ccell = np.repeat( np.arange( n ), nc )
    if "vertex_cuts" in cells._GEOMETRY:
        lvc = rows( cells.vertex_cuts )[ vmask ].astype( np.int64 ).reshape( -1, d )
    elif d == 2:                                         # vertex `i` is the corner of cuts `i - 1` and `i`
        li = np.arange( vp.shape[ 1 ] )[ None, : ]
        lvc = np.stack( [ ( li - 1 ) % np.maximum( nv[ :, None ], 1 ), np.broadcast_to( li, vmask.shape ) ], axis = -1 )[ vmask ]
    else:                                                # end `i` of a segment is cut `i`
        lvc = np.broadcast_to( np.arange( vp.shape[ 1 ] )[ None, :, None ], vmask.shape + ( 1, ) )[ vmask ]
    VC = lvc + coff[ vcell ][ :, None ]
    return V, vcell, VC, CID, ccell, coff


def _wall_of_cuts( V, VC, dom, walls ):
    """for each cut of the domain ( `dom`, a mask over the flat cuts ), the half-space of `walls` its vertices lie on: the one
    whose plane is the closest to the farthest of them"""
    dirs = np.asarray( walls[ 0 ], dtype = float ).reshape( len( walls[ 1 ] ), -1 )
    offs = np.asarray( walls[ 1 ], dtype = float ).reshape( -1 )
    nrm = np.maximum( np.linalg.norm( dirs, axis = 1 ), 1e-300 )
    dirs, offs = dirs / nrm[ :, None ], offs / nrm
    d = VC.shape[ 1 ] if VC.ndim == 2 else 1
    vrep = np.repeat( np.arange( len( VC ) ), d )
    cf = VC.reshape( -1 )
    sel = dom[ cf ]
    worst = np.zeros( ( len( dom ), len( offs ) ) )
    if sel.any():
        np.maximum.at( worst, cf[ sel ], np.abs( V[ vrep[ sel ] ] @ dirs.T - offs[ None, : ] ) )
    return np.argmin( worst[ dom ], axis = 1 ) if len( offs ) else np.zeros( int( dom.sum() ), np.int64 )


def _fit_walls( V, vcell, VC, dom, tol ):
    """the walls of a batch of cells that does not say them: the planes of their faces on the domain, each once"""
    d = V.shape[ 1 ]
    inside = np.zeros( ( int( vcell.max( initial = -1 ) ) + 1, d ) )
    np.add.at( inside, vcell, V )
    inside /= np.maximum( np.bincount( vcell, minlength = len( inside ) ), 1 )[ :, None ]
    planes = []
    for c in np.flatnonzero( dom ):
        on = np.flatnonzero( ( VC == c ).any( axis = 1 ) )
        if len( on ) < d:
            continue
        pts = V[ on ]
        n = np.linalg.svd( pts - pts.mean( axis = 0 ) )[ 2 ][ -1 ] if d > 1 else np.ones( 1 )
        o = float( n @ pts[ 0 ] )
        if n @ inside[ vcell[ on[ 0 ] ] ] > o:
            n, o = -n, -o
        planes.append( np.concatenate( [ n, [ o ] ] ) )
    if not planes:
        return np.zeros( ( 0, d ) ), np.zeros( 0 )
    planes = np.asarray( planes )
    q = np.round( planes / np.r_[ np.full( d, 1e-6 ), max( tol, 1e-300 ) ] ).astype( np.int64 )
    planes = planes[ _unique_rows( q, return_index = True )[ 1 ] ]
    return planes[ :, :d ], planes[ :, d ]


def _glue_vertices( W, vcell, VC, gen, img, fake_v, tol, quotient, walls ):
    """`( gv, E )`: the global number of each local vertex ( `-1` for a made-up one ) and the element of the group ( `[ nv, d, d + 1 ]` )
    from the frame of the vertex to that of its cell.

    THE KEY of a vertex: its `d + 1` generators -- the seed of its cell, and per cut the seed it faces with its image ( an element of
    the group ), or the plane of the wall it lies on --, expressed in the frame where the SMALLEST seed among them is itself ( its
    element the identity ): every cell that has the vertex says the same key, exactly. Where several images of that seed generate it
    ( a small torus ), the smallest key of the choices. Without a periodicity, the sorted generators.

    THE MERGE, where the diagram is not in general position ( more than `d + 1` cells at a point, a grid ): one point has several
    keys, which differ from cell to cell. Keys closer than `tol` are one vertex ( `_merge_close`, modulo the group ): the tolerance
    is a resolution, under which every cell collapses the same features."""
    nv_loc, d = VC.shape
    gv = np.full( nv_loc, -1, np.int64 )
    eye = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
    E = np.broadcast_to( eye, ( nv_loc, d, d + 1 ) ).copy()
    real = np.flatnonzero( ~fake_v )
    if len( real ) == 0:
        return gv, E
    rows, Ea, nseed = _canonical_rows( vcell[ real ], VC[ real ], gen, img, quotient, walls )
    if quotient is None:
        Ea = E[ real ]
    keys, inv, cnt = _unique_rows( rows, return_inverse = True, return_counts = True )
    nk = len( keys )

    # the point of each key, in its frame
    y = np.einsum( "mij,mj->mi", _inverse_maps( Ea )[ :, :, :d ], W[ real ] ) + _inverse_maps( Ea )[ :, :, d ]
    pts = np.zeros( ( nk, d ) )
    np.add.at( pts, inv, y )
    pts /= cnt[ :, None ]

    # the merge: closer than `tol` is one point, whatever the keys say ( a feature under the tolerance collapses in every cell alike )
    label = _merge_points( pts, tol, quotient, far_check = True )
    _, glob = np.unique( label, return_inverse = True )
    glob = glob.reshape( -1 )
    gv[ real ] = glob[ inv ]

    # the frame of a merged vertex: that of the first of its keys -- the others brought to it by an element of the group
    if quotient is not None:
        first_key = np.full( int( glob.max() ) + 1, -1, np.int64 )
        first_key[ glob[ ::-1 ] ] = np.arange( nk )[ ::-1 ]
        h = _relating_maps( pts, first_key[ glob ], quotient )                 # [ nk, d, d + 1 ]: y_k = h_k( y_first )
        Ea = _compose_maps( Ea, h[ inv ] )
    E[ real ] = Ea
    return gv, E


def _canonical_rows( own, cuts, gen, img, quotient, walls ):
    """`( rows, Ea, nseed )`: the KEY of each local face ( a vertex: `d` cuts ) from its generators -- the seed of its cell `own`, and
    per cut ( flat indices `cuts` ) the seed it faces with its image, or the plane of its wall -- expressed in the frame where the
    smallest seed among them is itself, the same from every cell that has the face ( the smallest key over the choices when several
    images of that seed generate it ). `Ea`: the element from that frame to the cell's; `nseed`: the number of seed generators ( a
    face met in each of them is COMPLETE ). Without a periodicity, the sorted generators."""
    m, c = cuts.shape
    seeds = np.concatenate( [ own[ :, None ], gen[ cuts ] ], axis = 1 )                     # [ m, c + 1 ]: >= 0 a seed, < 0 a wall
    nseed = ( seeds >= 0 ).sum( axis = 1 )
    if quotient is None:
        return np.sort( seeds, axis = 1 ), None, nseed
    G = quotient.ps.maps
    d = G.shape[ 1 ]
    els = np.concatenate( [ np.zeros( ( m, 1 ), np.int64 ), img[ cuts ] ], axis = 1 )
    planes = None if walls is None else _unit_planes( *walls )
    low = np.where( seeds >= 0, seeds, np.iinfo( np.int64 ).max ).min( axis = 1 )
    cands, oks, cand_E = [], [], []
    for a in range( c + 1 ):
        Ea = G[ els[ :, a ] ]
        inv = _inverse_maps( Ea )
        parts = []
        for r in range( c + 1 ):
            is_seed = seeds[ :, r ] >= 0
            key = _round_key( _compose_maps( inv, G[ els[ :, r ] ] ).reshape( m, -1 ) )
            if planes is not None and ( ~is_seed ).any():
                w = np.maximum( -1 - seeds[ :, r ], 0 )
                nrm, off = planes[ 0 ][ w ], planes[ 1 ][ w ]
                # the wall `n . x <= o`, `x = Ea y`: `( L^T n ) . y <= o - n . V`
                pk = _round_key( np.concatenate( [ np.einsum( "mji,mj->mi", Ea[ :, :, :d ], nrm ),
                                                   ( off - ( nrm * Ea[ :, :, d ] ).sum( axis = 1 ) )[ :, None ] ], axis = 1 ) )
                pk = np.concatenate( [ pk, np.zeros( ( m, key.shape[ 1 ] - pk.shape[ 1 ] ), np.int64 ) ], axis = 1 )
                key = np.where( is_seed[ :, None ], key, pk )
            parts.append( np.concatenate( [ np.where( is_seed, seeds[ :, r ], -1 )[ :, None ], key ], axis = 1 ) )
        P = np.stack( parts, axis = 1 )                                                         # [ m, c + 1, 1 + d ( d + 1 ) ]
        order = np.lexsort( P.transpose( 2, 0, 1 )[ ::-1 ] )                                    # ( the generators sorted, per face )
        P = np.take_along_axis( P, order[ :, :, None ], axis = 1 )
        cands.append( P.reshape( m, -1 ) )
        oks.append( ( seeds[ :, a ] >= 0 ) & ( seeds[ :, a ] == low ) )
        cand_E.append( Ea )
    allrows = np.concatenate( cands )
    _, rank = _unique_rows( allrows, return_inverse = True )
    rank = np.where( np.stack( oks ), rank.reshape( c + 1, m ), np.iinfo( np.int64 ).max )
    best = rank.argmin( axis = 0 )
    return allrows.reshape( c + 1, m, -1 )[ best, np.arange( m ) ], np.stack( cand_E )[ best, np.arange( m ) ], nseed


def _merge_points( pts, tol, quotient, far_check ):
    """the labels of `_merge_close` for points in their own frames: modulo the group with a periodicity"""
    if quotient is None:
        return _merge_close( pts, np.arange( len( pts ) ), len( pts ), tol, far_check = far_check )
    return _glue_points( pts, tol, quotient )


def _relating_maps( pts, ref, quotient ):
    """`[ m, d, d + 1 ]`: for each point `k`, an element `h` of the group with `pts[ k ] = h( pts[ ref[ k ] ] )` ( the identity for
    `ref[ k ] == k` ): both folded ( `PeriodicSetup.fold`, `y -> F y` ), then the element `a` of `maps` that brings one onto the other,
    `h = F_k^-1 a F_ref`"""
    ps = quotient.ps
    m, d = pts.shape
    Y, M, c = ps.fold( pts )
    Y = np.asarray( Y, dtype = float )
    M = np.broadcast_to( np.eye( d ), ( m, d, d ) ) if M is None else np.asarray( M, dtype = float ).reshape( m, d, d )
    F = np.concatenate( [ M, np.asarray( c, dtype = float ).reshape( m, d, 1 ) ], axis = 2 )
    G = ps.maps
    cand = np.einsum( "tij,mj->mti", G[ :, :, :d ], Y[ ref ] ) + G[ None, :, :, d ]
    a = np.argmin( ( ( cand - Y[ :, None, : ] ) ** 2 ).sum( axis = 2 ), axis = 1 )
    h = _compose_maps( _inverse_maps( F ), _compose_maps( G[ a ], F[ ref ] ) )
    same = ref == np.arange( m )
    h[ same ] = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
    return h


def _compose_maps( a, b ):
    """`a o b` for affine maps `[ ..., d, d + 1 ]`"""
    d = a.shape[ -2 ]
    return np.concatenate( [ a[ ..., :, :d ] @ b[ ..., :, :d ], ( a[ ..., :, :d ] @ b[ ..., :, d : ] ) + a[ ..., :, d : ] ], axis = -1 )


def _inverse_maps( a ):
    """the inverse of affine maps `[ ..., d, d + 1 ]` with ORTHOGONAL linear parts"""
    d = a.shape[ -2 ]
    Lt = np.swapaxes( a[ ..., :, :d ], -1, -2 )
    return np.concatenate( [ Lt, - Lt @ a[ ..., :, d : ] ], axis = -1 )


def _round_key( x ):
    """exact integer keys for the entries of elements of a discrete group ( and of planes ): rounded at `2^-20`"""
    return np.round( np.asarray( x, dtype = float ) * 2 ** 20 ).astype( np.int64 )


def _unit_planes( dirs, offs ):
    """the half-spaces `( n, o )` with unit normals"""
    nrm = np.maximum( np.linalg.norm( dirs, axis = 1 ), 1e-300 )
    return dirs / nrm[ :, None ], offs / nrm


def _same_place( tol, d ):
    """how far apart two copies of one merged vertex can be: the merge joins points that share a box of side `2 tol` ( `_merge_close` )"""
    return 2 * math.sqrt( d ) * tol * ( 1 + 1e-9 )


def _area_vectors( E ):
    """`[ m, d ]`: the area vectors of simplices of `d - 1` edges `E [ m, d - 1, d ]` -- `( -1 )^c det( E without column c ) / ( d - 1 )!`,
    the normal times the measure in any dimension"""
    m, k, d = E.shape
    res = np.zeros( ( m, d ) )
    for c in range( d ):
        M = np.delete( E, c, axis = 2 )
        res[ :, c ] = ( -1 ) ** c * ( np.linalg.det( M ) if k else np.ones( m ) )
    return res / math.factorial( k )


def _merge_ids( gv, pairs ):
    """`gv` with the global numbers of each pair made one ( the smallest ), renumbered"""
    m = int( gv.max( initial = -1 ) ) + 1
    lab = np.arange( m )
    pairs = np.asarray( pairs, dtype = np.int64 ).reshape( -1, 2 )
    while True:
        old = lab.copy()
        low = np.minimum( lab[ pairs[ :, 0 ] ], lab[ pairs[ :, 1 ] ] )
        np.minimum.at( lab, pairs[ :, 0 ], low )
        np.minimum.at( lab, pairs[ :, 1 ], low )
        lab = lab[ lab ]
        if ( lab == old ).all():
            break
    _, ren = np.unique( lab, return_inverse = True )
    return np.where( gv >= 0, ren.reshape( -1 )[ np.maximum( gv, 0 ) ], -1 )


def _glue_points( P, tol, quotient ):
    """the global number of each point, the points closer than `tol` MODULO THE GROUP being one: their representatives on the
    fundamental domain, and the images of those that lie on its border ( `_Quotient.canonical` )"""
    if len( P ) == 0:
        return np.zeros( 0, np.int64 )
    Y, owner, alias = quotient.canonical( P )
    label = _merge_close( np.concatenate( [ Y, alias ] ), np.concatenate( [ np.arange( len( P ) ), owner ] ), len( P ), tol,
                          far_check = len( owner ) == 0 )
    return np.unique( label, return_inverse = True )[ 1 ].reshape( -1 )


def _merge_close( pts, owners, m, tol, far_check ):
    """`[ m ]` labels: the points `owners` ( one point may have several rows: its images ) closer than `tol` share one -- they share
    a box of side `2 tol` for one of the `2^d` half-shifted grids, and labels flow through these boxes until they stop. `far_check`:
    a point farther than `tol` from the first of its group leaves it ( a chain of close points is not one point )"""
    d = pts.shape[ 1 ]
    lab = np.arange( m )
    if m < 2:
        return lab
    h = 2 * max( tol, 1e-300 )
    grids = [ _unique_rows( np.floor( pts / h + np.asarray( s ) ).astype( np.int64 ), return_inverse = True )[ 1 ]
              for s in itertools.product( ( 0.0, 0.5 ), repeat = d ) ]
    while True:
        old = lab.copy()
        for g in grids:
            low = np.full( int( g.max() ) + 1, m )
            np.minimum.at( low, g, lab[ owners ] )
            np.minimum.at( lab, owners, low[ g ] )
        lab = lab[ lab ]
        if ( lab == old ).all():
            break
    if far_check:
        far = np.abs( pts - pts[ lab ] ).max( axis = 1 ) > tol
        lab[ far ] = np.flatnonzero( far )
    return lab


class _Quotient:
    """THE QUOTIENT by the group of a periodicity, for the merge: `canonical( P )` gives the representatives of the points on the
    fundamental domain ( `PeriodicSetup.fold` ), and the images of those that the folding could have put elsewhere -- on the
    border of the parallelepiped of the lattice for translations, everywhere for a group with rotations or symmetries ( the
    folding then chooses among images that tie )"""

    def __init__( self, ps, tol ):
        self.ps, self.tol = ps, tol

    def canonical( self, P ):
        ps = self.ps
        Y = np.asarray( ps.fold( np.asarray( P, dtype = float ) )[ 0 ], dtype = float )
        B = ps.lattice
        if ps.translation_only and len( B ):
            # a point near the border of the parallelepiped of the lattice: its images across the borders it is near only ( a
            # translation by `+ b_r` near `k_r = 0`, `- b_r` near `k_r = 1`, and their sums ) -- not every element of the group
            Pi = np.linalg.pinv( B )
            k = ( Y - ps.anchor ) @ Pi
            fr = k - np.floor( k )
            lim = 4 * self.tol * np.linalg.norm( Pi, axis = 0 )
            lo, hi = fr < lim, fr > 1 - lim
            near = lo | hi
            idx = np.flatnonzero( near.any( axis = 1 ) )
            if len( idx ) == 0:
                return Y, np.zeros( 0, np.int64 ), np.zeros( ( 0, Y.shape[ 1 ] ) )
            step = np.where( lo[ idx ], 1.0, -1.0 ) * near[ idx ]                  # [ m, r ]: the move along each lattice vector
            r = B.shape[ 0 ]
            owners, alias = [], []
            for mask in range( 1, 2 ** r ):
                bits = np.array( [ ( mask >> i ) & 1 for i in range( r ) ], bool )
                use = ( near[ idx ][ :, bits ] ).all( axis = 1 )
                if not use.any():
                    continue
                sh = ( step[ use ][ :, bits ] ) @ B[ bits ]
                owners.append( idx[ use ] )
                alias.append( Y[ idx[ use ] ] + sh )
            return Y, np.concatenate( owners ), np.concatenate( alias )
        else:
            flag = np.ones( len( Y ), bool )
        idx = np.flatnonzero( flag )
        maps = ps.maps[ 1: ]
        if len( idx ) == 0 or len( maps ) == 0:
            return Y, np.zeros( 0, np.int64 ), np.zeros( ( 0, Y.shape[ 1 ] ) )
        d = Y.shape[ 1 ]
        alias = np.concatenate( [ Y[ idx ] @ a[ :, :d ].T + a[ :, d ] for a in maps ] )
        return Y, np.tile( idx, len( maps ) ), alias


def _moved( W, vcell, maps ):
    """the vertices of each cell moved by its affine map `maps[ cell ]` ( `[ n, d, d + 1 ]` )"""
    d = W.shape[ 1 ]
    A = np.asarray( maps, dtype = float )[ vcell ]
    return np.einsum( "vij,vj->vi", A[ :, :, :d ], W ) + A[ :, :, d ]


def _unique_rows( A, return_index = False, return_inverse = False, return_counts = False ):
    """`np.unique( A, axis = 0 )` for integer rows, without sorting rows as records ( 20 times slower ): each row packed into ONE
    int64 when the ranges of the columns allow it ( lexicographic order kept: the first column the most significant ), a
    `lexsort` otherwise"""
    A = np.asarray( A, np.int64 ).reshape( len( A ), -1 )
    m, w = A.shape
    if m == 0:
        res = [ A ] + [ np.zeros( 0, np.int64 ) ] * ( return_index + return_inverse + return_counts )
        return tuple( res ) if len( res ) > 1 else res[ 0 ]
    lo = A.min( axis = 0 )
    span = A.max( axis = 0 ) - lo + 1
    if np.sum( np.log2( span.astype( float ) ) ) < 62:
        mult = np.concatenate( [ np.cumprod( span[ ::-1 ] )[ ::-1 ][ 1: ], [ 1 ] ] )
        key = ( ( A - lo ) * mult ).sum( axis = 1 )
        order = np.argsort( key, kind = "stable" )
        k = key[ order ]
        first = np.concatenate( [ [ True ], k[ 1: ] != k[ :-1 ] ] )
    else:
        order = np.lexsort( A.T[ ::-1 ] )
        k = A[ order ]
        first = np.concatenate( [ [ True ], ( k[ 1: ] != k[ :-1 ] ).any( axis = 1 ) ] )
    starts = np.flatnonzero( first )
    res = [ A[ order[ starts ] ] ]
    if return_index:
        res.append( order[ starts ] )
    if return_inverse:
        inv = np.empty( m, np.int64 )
        inv[ order ] = np.cumsum( first ) - 1
        res.append( inv )
    if return_counts:
        res.append( np.diff( np.concatenate( [ starts, [ m ] ] ) ) )
    return tuple( res ) if len( res ) > 1 else res[ 0 ]


# ---- arrays of pairs -------------------------------------------------------------------------------

def _unique_pairs( a, b ):
    """`[ m, 2 ]`, the distinct pairs `( a, b )`, sorted"""
    if len( a ) == 0:
        return np.zeros( ( 0, 2 ), np.int64 )
    return _unique_rows( np.stack( [ np.asarray( a, np.int64 ), np.asarray( b, np.int64 ) ], axis = 1 ) )


def _csr( pairs, m ):
    """`( offsets [ m + 1 ], ids )` of sorted pairs"""
    if len( pairs ) == 0:
        return np.zeros( m + 1, np.int64 ), np.zeros( 0, np.int64 )
    pairs = pairs[ np.lexsort( ( pairs[ :, 1 ], pairs[ :, 0 ] ) ) ]
    off = np.concatenate( [ [ 0 ], np.cumsum( np.bincount( pairs[ :, 0 ], minlength = m ) ) ] )
    return off, pairs[ :, 1 ].copy()


def _compose( ab, bc, nb_b ):
    """the pairs `( a, c )` of `( a, b )` and `( b, c )`"""
    off, ids = _csr( bc, nb_b )
    cnt = np.diff( off )[ ab[ :, 1 ] ]
    a = np.repeat( ab[ :, 0 ], cnt )
    starts = np.repeat( off[ ab[ :, 1 ] ], cnt )
    c = ids[ starts + np.arange( len( a ) ) - np.repeat( np.cumsum( cnt ) - cnt, cnt ) ]
    return _unique_pairs( a, c )


def _bases( P, off, ids, k ):
    """`[ m, k, d ]`: orthonormal vectors of the points of each group ( `ids[ off[ i ] : off[ i + 1 ] ]` ): the first `j` span the affine
    hull of a `j`-dimensional group, the last ones ( `k = d` ) its normals. In closed form for what the complex asks in 2D and 3D --
    the direction of an edge, the plane of a polygon, the normal of a facet: the farthest point from the first, then the farthest
    from their line --, by a batched SVD otherwise."""
    m, d = len( off ) - 1, P.shape[ 1 ]
    if d not in ( 2, 3 ) or m == 0 or len( ids ) == 0:
        return _bases_svd( P, off, ids, k )
    cnt = np.diff( off )
    owner = np.repeat( np.arange( m ), cnt )
    p0 = P[ ids[ np.minimum( off[ :-1 ], len( ids ) - 1 ) ] ]
    r = P[ ids ] - p0[ owner ]
    def far( score ):                                    # the point of each group with the largest score
        order = np.lexsort( ( -score, owner ) )
        first = np.full( m, -1, np.int64 )
        first[ owner[ order ][ ::-1 ] ] = order[ ::-1 ]
        return first
    f1 = far( ( r ** 2 ).sum( axis = 1 ) )
    u = np.where( ( f1 >= 0 )[ :, None ], r[ np.maximum( f1, 0 ) ], 0.0 )
    nu = np.linalg.norm( u, axis = 1 )
    u = np.where( ( nu > 0 )[ :, None ], u / np.maximum( nu, 1e-300 )[ :, None ], np.eye( d )[ 0 ] )
    if d == 2:
        n = np.stack( [ u[ :, 1 ], - u[ :, 0 ] ], axis = 1 )
        res = np.stack( [ u, n ], axis = 1 )
    else:
        f2 = far( ( np.cross( u[ owner ], r ) ** 2 ).sum( axis = 1 ) )
        w = np.cross( u, np.where( ( f2 >= 0 )[ :, None ], r[ np.maximum( f2, 0 ) ], 0.0 ) )
        nw = np.linalg.norm( w, axis = 1 )
        # ( a group on a line: any normal to it )
        alt = np.cross( u, np.where( np.abs( u[ :, :1 ] ) < 0.9, np.eye( 3 )[ 0 ], np.eye( 3 )[ 1 ] ) )
        w = np.where( ( nw > 1e-300 )[ :, None ], w / np.maximum( nw, 1e-300 )[ :, None ], alt / np.linalg.norm( alt, axis = 1, keepdims = True ) )
        v = np.cross( w, u )
        res = np.stack( [ u, v, w ], axis = 1 )
    return res[ :, :k ] if k < d else res


def _bases_svd( P, off, ids, k ):
    """`[ m, k, d ]`: the right singular vectors of the points of each group ( `ids[ off[ i ] : off[ i + 1 ] ]` ), centered --
    the first `j` span the affine hull of a `j`-dimensional group, the last ones its normals. Batched: each group padded with
    its first point, which spans nothing more."""
    m, d = len( off ) - 1, P.shape[ 1 ]
    cnt = np.diff( off )
    L = int( cnt.max( initial = 1 ) )
    idx = np.where( np.arange( L )[ None, : ] < cnt[ :, None ], off[ :-1, None ] + np.arange( L )[ None, : ], off[ :-1, None ] )
    idx = np.minimum( idx, max( len( ids ) - 1, 0 ) )
    pts = P[ ids[ idx ] ] if len( ids ) else np.zeros( ( m, L, d ) )
    w = ( np.arange( L )[ None, : ] < cnt[ :, None ] )[ :, :, None ]
    mean = ( pts * w ).sum( axis = 1 ) / np.maximum( cnt, 1 )[ :, None ]
    vt = np.linalg.svd( ( pts - mean[ :, None, : ] ) * w, full_matrices = True )[ 2 ]
    return vt[ :, :k ] if k < d else vt
