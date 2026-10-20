"""PERIODICITY: transport on the quotient of the space by a group of ISOMETRIES ( a torus, a cylinder, a Moebius strip,
a Klein bottle, a crystal... ).

    sdot.plan( f, g, periodicity = True )                          # every side of the axis-aligned box of `g`
    sdot.plan( f, g, periodicity = [ True, False ] )               # per axis ( axis-aligned box ): a cylinder
    sdot.plan( f, g, periodicity = [ [ 1, 0 ], [ 0.5, 1 ] ] )      # the generators of the group: translations...
    sdot.plan( f, g, periodicity = [ ( [ [ 1, 0 ], [ 0, -1 ] ], [ 1, 1 ] ) ] )   # ... or any isometry `x -> M x + V`
    sdot.plan( f, g, periodicity = g.domain )                      # the opposite parallel facets of the support, paired

What is described is the GROUP, not a domain: each generator is an isometry, given as a vector ( a translation ), a pair
`( M, V )` ( the map `x -> M x + V`, `M` orthogonal ) or a `( d + 1 )²` homogeneous matrix. Their inverses and their
compositions are added here. The support of the target must lie in a fundamental domain: its images by the group do not
overlap.

= Internals

THE CONSTRUCTION ( `diagram/Common.h`, `cell/Providers.h`, `Periodic.h` ). One cell per seed, built in the plane around
its seed ( unfolded ):

  * the seeds are FOLDED onto the support ( `PeriodicSetup.fold` ): reduced by the translations of the group, then moved by
    the element of the group that brings them closest to the support. The results come back in the frame of each seed: the
    barycenters are unfolded by the inverse map ( `PeriodicSetup.unfold` );
  * a cell cannot leave the Dirichlet cell of its seed among its own images ( same weight: their bisector is the
    ordinary one ), whatever the weights -- in particular among its images by the TRANSLATIONS of the group, a lattice.
    The start cell is therefore the box of the folded seeds and of the support, widened by the half-extent `e` of the
    Dirichlet cell of that lattice ( exactly half a period for a rectangular lattice ); along the axes the lattice does not
    span, the box of the images of the support that are kept;
  * the planes are those of the seeds and of their images `j + n t`, at `L_t positions( j ) + s_t`: the tree is traversed
    once per image, its boxes moved. An image can cut a cell only if its box comes within `2 e` of the region of the seeds
    ( the nearest image of a seed is within `e` of the point, the point within `e` of its seed );
  * the density is the sum of its images by the same maps, each cutting the cell into its own pieces.
"""

import numpy as np


# ---- the description ------------------------------------------------------------------------------

def _as_isometry( g, d ):
    """one generator as an affine map `[ d, d + 1 ]` ( `[ M | V ]` ): a vector, a `( M, V )` pair or a `( d + 1 )²` matrix"""
    if isinstance( g, ( tuple, list ) ) and len( g ) == 2 and np.ndim( g[ 0 ] ) == 2:
        M, V = np.asarray( g[ 0 ], dtype = float ), np.asarray( g[ 1 ], dtype = float ).reshape( -1 )
    else:
        a = np.asarray( g, dtype = float )
        if a.ndim == 1:
            if a.shape[ 0 ] != d:
                raise ValueError( f"periodicity: a translation of { a.shape[ 0 ] } components in { d }D" )
            M, V = np.eye( d ), a
        elif a.shape == ( d + 1, d + 1 ):
            M, V = a[ :d, :d ], a[ :d, d ]
        else:
            raise ValueError( f"periodicity: a generator is a vector [ { d } ], a pair ( M, V ) or a [ { d + 1 }, { d + 1 } ] matrix ( got { a.shape } )" )
    if M.shape != ( d, d ) or V.shape != ( d, ):
        raise ValueError( f"periodicity: ( M, V ) with M [ { d }, { d } ] and V [ { d } ] ( got { M.shape } and { V.shape } )" )
    if not np.allclose( M.T @ M, np.eye( d ), atol = 1e-10 ):
        raise ValueError( "periodicity: the linear part of a transformation must be orthogonal ( an isometry ): otherwise the "
                          "transport cost is not invariant and the quotient means nothing" )
    return np.concatenate( [ M, V[ :, None ] ], axis = 1 )


def _pairs_of_facets( dirs, offs ):
    """the translations that pair the opposite parallel facets of a support given as half-spaces"""
    dirs = np.asarray( dirs, dtype = float )
    offs = np.asarray( offs, dtype = float ).reshape( -1 )
    nrm = np.linalg.norm( dirs, axis = 1 )
    u, o = dirs / nrm[ :, None ], offs / nrm
    res, used = [], set()
    for a in range( len( o ) ):
        if a in used:
            continue
        for b in range( a + 1, len( o ) ):
            if b not in used and np.allclose( u[ a ], - u[ b ], atol = 1e-12 ):
                # the slab `- o_b <= u_a . x <= o_a`: the translation that sends the facet `b` onto the facet `a`
                res.append( u[ a ] * ( o[ a ] + o[ b ] ) )
                used.update( ( a, b ) )
                break
        else:
            raise ValueError( "periodicity: a facet of the domain has no opposite parallel facet to be paired with" )
    return np.array( res )


def generators_of( periodicity, d, support = None ):
    """THE GENERATORS `[ r, d, d + 1 ]` ( affine isometries `[ M | V ]` ) described by `periodicity`, or `None`. `support`:
    the half-spaces of the support of the target, which `True` and the per-axis booleans read."""
    if periodicity is None or periodicity is False:
        return None

    def translations( vs ):
        vs = np.asarray( vs, dtype = float ).reshape( -1, d )
        return np.concatenate( [ np.broadcast_to( np.eye( d ), ( len( vs ), d, d ) ), vs[ :, :, None ] ], axis = 2 ) if len( vs ) else None

    def box():
        from .PowerDiagram import axis_aligned_box
        b = None if support is None else axis_aligned_box( np.asarray( support[ 0 ], dtype = float ), np.asarray( support[ 1 ], dtype = float ) )
        if b is None or b[ 2 ].any():
            raise ValueError( "periodicity = True ( or one boolean per axis ) reads the sides of an AXIS-ALIGNED box: the support of "
                              "this target is not one -- give the generators of the group, `( M, V )` pairs or matrices, which glue "
                              "any domain" )
        return np.asarray( b[ 0 ], dtype = float ), np.asarray( b[ 1 ], dtype = float )

    if periodicity is True:
        lo, hi = box()
        return translations( np.diag( hi - lo ) )

    if isinstance( periodicity, ( tuple, list ) ) and len( periodicity ) and all( isinstance( p, ( bool, np.bool_ ) ) for p in periodicity ):
        if len( periodicity ) != d:
            raise ValueError( f"periodicity: { len( periodicity ) } booleans in { d }D" )
        lo, hi = box()
        return translations( np.diag( hi - lo )[ np.asarray( periodicity, dtype = bool ) ] )

    # half-spaces `( directions, offsets )`: `g.domain`
    if isinstance( periodicity, tuple ) and len( periodicity ) == 2 and np.ndim( periodicity[ 0 ] ) == 2 and np.ndim( periodicity[ 1 ] ) == 1 \
            and np.shape( periodicity[ 0 ] ) == ( len( periodicity[ 1 ] ), d ):
        return translations( _pairs_of_facets( *periodicity ) )

    gens = [ _as_isometry( g, d ) for g in periodicity ]
    if not gens:
        return None
    gens = np.array( gens )
    if np.allclose( gens[ :, :, :d ], np.eye( d ) ) and np.linalg.matrix_rank( gens[ :, :, d ] ) != len( gens ):
        raise ValueError( "periodicity: translations that are not linearly independent ( one per direction: inverses and combinations are added )" )
    return gens


# ---- the group -----------------------------------------------------------------------------------

def _compose( a, b ):
    """`a o b`, affine maps `[ d, d + 1 ]`"""
    d = a.shape[ 0 ]
    return np.concatenate( [ a[ :, :d ] @ b[ :, :d ], ( a[ :, :d ] @ b[ :, d ] + a[ :, d ] )[ :, None ] ], axis = 1 )


def _inverse( a ):
    d = a.shape[ 0 ]
    Mt = a[ :, :d ].T
    return np.concatenate( [ Mt, ( - Mt @ a[ :, d ] )[ :, None ] ], axis = 1 )


def _image_box( a, lo, hi ):
    """the box of the image of the box `[ lo, hi ]` by the affine map `a`"""
    d = a.shape[ 0 ]
    c, h = ( lo + hi ) / 2, ( hi - lo ) / 2
    cc = a[ :, :d ] @ c + a[ :, d ]
    hh = np.abs( a[ :, :d ] ) @ h
    return cc - hh, cc + hh


def _key( a, scale ):
    return tuple( np.round( a.reshape( -1 ) / scale * 1e8 ).astype( np.int64 ) )


def _lattice_basis( vectors, d ):
    """a basis of the lattice spanned by `vectors` ( the shortest independent ones, which are a basis in 1D, 2D and 3D ),
    `[ r, d ]`"""
    vs = sorted( ( v for v in vectors if np.linalg.norm( v ) > 0 ), key = np.linalg.norm )
    basis = []
    for v in vs:
        if np.linalg.matrix_rank( np.array( basis + [ v ] ), tol = 1e-9 * np.linalg.norm( v ) ) > len( basis ):
            basis.append( v )
        if len( basis ) == d:
            break
    return np.array( basis ).reshape( -1, d )


def support_box( support, d ):
    """the bounding box `( lo, hi )` of half-spaces"""
    from .PowerDiagram import axis_aligned_box
    dirs = np.asarray( support[ 0 ], dtype = float ).reshape( -1, d )
    offs = np.asarray( support[ 1 ], dtype = float ).reshape( -1 )
    b = axis_aligned_box( dirs, offs )
    if b is not None and not b[ 2 ].any():
        return np.asarray( b[ 0 ], dtype = float ), np.asarray( b[ 1 ], dtype = float )
    from .Cell import Cell
    dom = Cell.make_unbounded( d, kernel_dtype = "FP64" )
    for k in range( len( offs ) ):
        dom.cut( dirs[ k ], float( offs[ k ] ) )
    if not dom.is_bounded:
        raise ValueError( "periodicity: the support of the target does not bound the domain" )
    v = np.asarray( dom.vertices(), dtype = float ).reshape( -1, d )
    return v.min( axis = 0 ), v.max( axis = 0 )


class PeriodicSetup:
    """What a diagram needs to build the cells of the quotient by the group of `generators` ( `[ r, d, d + 1 ]` ), from the
    support of the target ( see the module docstring ):

      * `maps` ( `[ nb_images, d, d + 1 ]`, the identity first ): the elements of the group whose images can cut a cell or
        carry density into one -- `shifts` and `linear` ( `None` for translations only ) are their two parts;
      * `lattice`: a basis of the translations of the group;
      * `start_lo / start_hi`: the start cell; `boundaries`: the half-spaces of the support that the group leaves in place.

    It does not depend on the positions: `fold` brings them onto the support, `unfold` takes points back."""

    MAX_IMAGES = 4096

    def __init__( self, generators, support, seed_box = None ):
        G = np.asarray( generators, dtype = float )
        r, d = G.shape[ 0 ], G.shape[ 1 ]
        dirs = np.asarray( support[ 0 ], dtype = float ).reshape( -1, d )
        offs = np.asarray( support[ 1 ], dtype = float ).reshape( -1 )
        lo, hi = support_box( ( dirs, offs ), d )
        self.nb_dims = d
        self.anchor, self.support_lo, self.support_hi = lo, lo, hi
        self.support = ( dirs, offs )
        self.generators = G
        scale = max( float( np.abs( hi - lo ).max() ), 1e-300 )
        eye = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
        moves = [ g for g in G ] + [ _inverse( g ) for g in G ]
        self.translation_only = bool( np.allclose( G[ :, :, :d ], np.eye( d ) ) )

        # THE TRANSLATIONS of the group: the generators themselves for a lattice, otherwise found among the short words
        if self.translation_only:
            B = G[ :, :, d ]
        else:
            seen, front, trans = { _key( eye, scale ) }, [ eye ], []
            for _ in range( 6 ):
                nxt = []
                for a in front:
                    for m in moves:
                        b = _compose( m, a )
                        k = _key( b, scale )
                        if k in seen:
                            continue
                        seen.add( k )
                        nxt.append( b )
                        if np.allclose( b[ :, :d ], np.eye( d ), atol = 1e-12 ):
                            trans.append( b[ :, d ] )
                front = nxt
            B = _lattice_basis( trans, d )
        self.lattice = B

        # the region of the folded seeds and of the support
        if len( B ):
            plo, phi = lo + np.minimum( B, 0 ).sum( axis = 0 ), lo + np.maximum( B, 0 ).sum( axis = 0 )
            rlo, rhi = np.minimum( lo, plo ), np.maximum( hi, phi )
        else:
            rlo, rhi = lo.copy(), hi.copy()
        if seed_box is not None:                        # folded seeds that the group cannot bring onto the support ( a cone )
            rlo, rhi = np.minimum( rlo, seed_box[ 0 ] ), np.maximum( rhi, seed_box[ 1 ] )
        self.region_lo, self.region_hi = rlo, rhi

        # the half-extent of the Dirichlet cell of the lattice, per axis: half a period for a rectangular lattice, the
        # covering radius bound `sqrt( sum |b_i|² ) / 2` otherwise, on the axes the lattice spans
        if len( B ):
            nz = np.abs( B ) > 1e-14 * np.abs( B ).max()
            if ( nz.sum( axis = 1 ) == 1 ).all() and ( nz.sum( axis = 0 ) <= 1 ).all():
                e = np.abs( B ).sum( axis = 0 ) / 2
            else:
                e = np.where( nz.any( axis = 0 ), np.sqrt( ( B ** 2 ).sum() ) / 2, 0.0 )
            spanned = nz.any( axis = 0 )
        else:
            e, spanned = np.zeros( d ), np.zeros( d, dtype = bool )

        # THE IMAGES: the elements of the group whose image of the region comes within `2 e` of it ( strictly: a tie cuts
        # nothing ). Found by a walk on the words, through the elements that stay within twice that distance.
        def near( a, margin ):
            blo, bhi = _image_box( a, rlo, rhi )
            return ( ( blo < rhi + margin ) & ( bhi > rlo - margin ) ).all()
        # ( along the axes the lattice spans, a tie at `2 e` cuts nothing; along the others, a finite group's images that only
        # touch the region -- a rotation about its corner -- do cut )
        lim = np.where( spanned, 2 * e * ( 1 - 1e-12 ), 1e-9 * scale )
        maps, seen, front = [ eye ], { _key( eye, scale ) }, [ eye ]
        reach = 2 * lim + ( rhi - rlo )
        while front:
            nxt = []
            for a in front:
                for m in moves:
                    b = _compose( m, a )
                    k = _key( b, scale )
                    if k in seen or not near( b, reach ):
                        continue
                    seen.add( k )
                    nxt.append( b )
                    if near( b, lim ):
                        maps.append( b )
                    if len( seen ) > self.MAX_IMAGES:
                        raise ValueError( "periodicity: the group has too many elements near the support -- is it discrete, "
                                          "and is the support a fundamental domain of it?" )
            front = nxt
        self.maps = np.array( maps )
        self.shifts = self.maps[ :, :, d ]
        self.linear = None if self.translation_only else self.maps[ :, :, :d ]

        # the start cell: the region widened by `e` along the axes the lattice spans; along the others, the box of the images
        # of the region that are kept ( a finite group: the cells may turn around )
        blo, bhi = rlo.copy(), rhi.copy()
        for a in self.maps:
            ilo, ihi = _image_box( a, rlo, rhi )
            blo, bhi = np.minimum( blo, ilo ), np.maximum( bhi, ihi )
        self.start_lo = np.where( spanned, rlo - e, blo )
        self.start_hi = np.where( spanned, rhi + e, bhi )

        # the support of the target must lie in a fundamental domain: the images of its box do not overlap it
        vol = np.prod( hi - lo )
        if self.translation_only and len( B ) == d and vol > abs( np.linalg.det( B ) ) * ( 1 + 1e-9 ):
            raise ValueError( "periodicity: the support of the target is larger than a period ( its box has a volume of "
                              f"{ vol :g }, a period { abs( np.linalg.det( B ) ) :g } ): its images would overlap" )

        # the half-spaces of the support that stay WALLS of the quotient: those that every image of the support stays behind ( a
        # cylinder keeps the walls of its free axis, a Moebius strip its two walls, a cone the outer sides of its quarter )
        nrm = np.linalg.norm( dirs, axis = 1 )
        u, o = dirs / nrm[ :, None ], offs / nrm
        corners = np.array( [ np.where( c, hi, lo ) for c in np.ndindex( *( ( 2, ) * d ) ) ], dtype = float )
        images = np.concatenate( [ corners @ a[ :, :d ].T + a[ :, d ] for a in self.maps ] )
        keep = ( images @ u.T <= o[ None, : ] + 1e-9 * scale ).all( axis = 0 )
        self.boundaries = ( dirs[ keep ], offs[ keep ] )

    def maps_meeting( self, lo, hi ):
        """the elements of the group whose image of the box `[ lo, hi ]` meets the start box, as affine maps `[ m, d, d + 1 ]`,
        the identity first: where a density whose mass lies in that box ( a gaussian and its tails ) has images that a cell can
        see ( `Distribution.periodized` ). Found by a walk on the words, through the elements whose image stays within the size
        of the two boxes."""
        d = self.nb_dims
        lo, hi = np.asarray( lo, dtype = float ), np.asarray( hi, dtype = float )
        slo, shi = self.start_lo, self.start_hi
        scale = max( float( np.abs( shi - slo ).max() ), 1e-300 )
        reach = ( hi - lo ) + ( shi - slo )
        def near( a, margin ):
            blo, bhi = _image_box( a, lo, hi )
            return ( ( blo < shi + margin ) & ( bhi > slo - margin ) ).all()
        eye = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
        moves = [ g for g in self.generators ] + [ _inverse( g ) for g in self.generators ]
        maps, seen, front = [ eye ], { _key( eye, scale ) }, [ eye ]
        while front:
            nxt = []
            for a in front:
                for m in moves:
                    b = _compose( m, a )
                    k = _key( b, scale )
                    if k in seen or not near( b, reach ):
                        continue
                    seen.add( k )
                    nxt.append( b )
                    if near( b, 0.0 ):
                        maps.append( b )
                    if len( seen ) > self.MAX_IMAGES:
                        raise ValueError( "periodicity: the group has too many elements near the support of the density -- is it "
                                          "discrete, and is the support a fundamental domain of it?" )
            front = nxt
        return np.array( maps )

    # ---- the rings -----------------------------------------------------------------------------------

    def support_vertices( self ):
        """the vertices of the support ( a convex polytope ), `[ m, d ]`"""
        from .Cell import Cell
        d = self.nb_dims
        dom = Cell.make_unbounded( d, kernel_dtype = "FP64" )
        for n, o in zip( *self.support ):
            dom.cut( n, float( o ) )
        return np.asarray( dom.vertices(), dtype = float ).reshape( -1, d )

    def ring( self, k = 1 ):
        """THE ELEMENTS OF THE GROUP UP TO THE RING `k`, as affine maps `[ m, d, d + 1 ]`, the identity first: ring 1 holds those
        whose image of the support touches the support ( the 8 neighbours of a square on a torus, turned or flipped as the group
        says ), ring `k` those whose image touches an image of ring `k - 1`."""
        d = self.nb_dims
        verts = self.support_vertices()
        scale = max( float( np.ptp( verts, axis = 0 ).max() ), 1e-300 )
        tol = 1e-9 * scale
        eye = np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )
        moves = [ g for g in self.generators ] + [ _inverse( g ) for g in self.generators ]

        def box( a ):
            q = verts @ a[ :, :d ].T + a[ :, d ]
            return q.min( axis = 0 ), q.max( axis = 0 )

        def touches( a, boxes ):
            lo, hi = box( a )
            return any( ( lo <= bh + tol ).all() and ( hi >= bl - tol ).all() for bl, bh in boxes )

        res, seen = [ eye ], { _key( eye, scale ) }
        for _ in range( int( k ) ):
            prev = [ box( a ) for a in res ]
            queue = list( res )
            while queue:
                a = queue.pop()
                for m in moves:
                    b = _compose( m, a )
                    key = _key( b, scale )
                    if key in seen or not touches( b, prev ):
                        continue
                    seen.add( key )
                    res.append( b )
                    queue.append( b )
        return np.array( res )

    # ---- the seeds ---------------------------------------------------------------------------------

    def fold( self, positions ):
        """`( folded, M, c )`: the positions brought onto the support, `folded = M positions + c` per seed ( `M` `[ n, d, d ]`, `c`
        `[ n, d ]`, or `M = None` for translations only ). A backend array ( maybe traced ) gives backend arrays: the map is
        chosen on the detached positions, and the derivative of `folded` is that of the map applied to `positions`."""
        d = self.nb_dims
        B = self.lattice
        numpy_in = isinstance( positions, np.ndarray ) or not hasattr( positions, "shape" )
        if numpy_in:
            import types
            ops = types.SimpleNamespace(
                stop_gradient = lambda x: x, array = lambda a, dt: np.asarray( a, dtype = float ), dtype_of = lambda x: float,
                matmul = np.matmul, where = np.where, clip = np.clip, sum = lambda a, axis = None: np.sum( a, axis = axis ) )
            positions = np.asarray( positions, dtype = float )
        else:
            import loom
            ops = loom.ops()
        dt = ops.dtype_of( positions )

        # the translations of the group: the parallelepiped of the lattice, anchored at the corner of the support
        if len( B ):
            P = np.linalg.pinv( B )                      # [ d, r ]: `c = ( x - anchor ) P`
            x = ops.stop_gradient( positions ) - ops.array( self.anchor, dt )
            k = ops.matmul( x, ops.array( P, dt ) )
            shift = ops.matmul( k - k % 1, ops.array( B, dt ) )
        else:
            shift = 0 * positions
        y = positions - shift
        if self.translation_only:
            return y, None, - shift

        # then the element of the group that brings it closest to the support
        lo, hi = ops.array( self.support_lo, dt ), ops.array( self.support_hi, dt )
        best, best_M, best_c, best_dist = y, None, - shift, None
        n = y.shape[ 0 ]
        for a in self.maps:
            M, V = a[ :, :d ], a[ :, d ]
            z = ops.matmul( y, ops.array( M.T, dt ) ) + ops.array( V, dt )
            zs = ops.stop_gradient( z )
            gap = zs - ops.clip( zs, lo, hi )
            dist = ops.sum( gap * gap, axis = 1 )
            Mb = ops.array( np.broadcast_to( M, ( n, d, d ) ).copy(), dt )
            cb = ops.matmul( - shift, ops.array( M.T, dt ) ) + ops.array( V, dt )
            if best_dist is None:
                best, best_M, best_c, best_dist = z, Mb, cb, dist
            else:
                better = dist < best_dist * ( 1 - 1e-12 ) - 1e-300
                best = ops.where( better[ :, None ], z, best )
                best_M = ops.where( better[ :, None, None ], Mb, best_M )
                best_c = ops.where( better[ :, None ], cb, best_c )
                best_dist = ops.where( better, dist, best_dist )
        return best, best_M, best_c

    @staticmethod
    def unfold( points, M, c ):
        """the points of the folded frame taken back to the frame of the given positions: `M^T ( points - c )`"""
        if M is None:
            return points - c
        q = points - c
        return ( M * q[ :, :, None ] ).sum( axis = 1 ) if isinstance( q, np.ndarray ) else _ops_unfold( q, M )


def _ops_unfold( q, M ):
    import loom
    return loom.ops().sum( M * q[ :, :, None ], axis = 1 )


def setup_for( positions, boundaries, periodicity ):
    """`( setup, folded )` for a diagram of these positions ( a backend array, maybe traced ) on the domain `boundaries`
    ( half-spaces ), or `( None, positions )` without a periodicity. The setup carries the folding ( `fold_M`, `fold_c` ). A
    group that cannot bring every seed onto the support ( a cone ) has the region of its cells widened to the folded seeds,
    when they are concrete ( under a trace they are assumed on the support )."""
    import loom
    d = int( positions.shape[ 1 ] )
    gens = generators_of( periodicity, d, boundaries )
    if gens is None:
        return None, positions
    if d < 2:
        raise ValueError( "periodicity: 2D and above ( the circle is another algorithm )" )
    ps = PeriodicSetup( gens, boundaries )
    folded, ps.fold_M, ps.fold_c = ps.fold( positions )
    if not loom.is_traced( folded ) and not ps.translation_only:
        f = np.asarray( folded, dtype = float ).reshape( -1, d )
        if len( f ) and ( ( f.min( axis = 0 ) < ps.region_lo ) | ( f.max( axis = 0 ) > ps.region_hi ) ).any():
            ps = PeriodicSetup( gens, boundaries, seed_box = ( f.min( axis = 0 ), f.max( axis = 0 ) ) )
            folded, ps.fold_M, ps.fold_c = ps.fold( positions )
    return ps, folded


def wrap( positions, periodicity, target = None ):
    """The `positions` brought back onto the support of `target` by the group that `periodicity` describes ( onto the
    parallelepiped of its translations anchored at the origin, without a target ).

    = Internals

    What the solver does to the seeds before building the cells ( `PeriodicSetup.fold` ); the results come back unfolded."""
    pos = np.asarray( positions, dtype = float )
    d = pos.shape[ -1 ]
    support = None if target is None else target.periodic_support()
    gens = generators_of( periodicity, d, support )
    if gens is None:
        return pos
    if support is None:
        support = ( np.concatenate( [ np.eye( d ), - np.eye( d ) ] ), np.zeros( 2 * d ) )
        if not np.allclose( gens[ :, :, :d ], np.eye( d ) ):
            raise ValueError( "wrap: a group with rotations or symmetries folds onto the support of a target ( give `target` )" )
        B = gens[ :, :, d ]
        P = np.linalg.pinv( B )
        return ( pos.reshape( -1, d ) - np.floor( pos.reshape( -1, d ) @ P ) @ B ).reshape( pos.shape )
    return PeriodicSetup( gens, support ).fold( pos.reshape( -1, d ) )[ 0 ].reshape( pos.shape )
