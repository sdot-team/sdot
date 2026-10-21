"""GROUPS OF SEEDS: a tree of AGGREGATES, seeds that share one cell and split it among themselves ( `PowerDiagram( groups = ... )` ).

Two seeds at the SAME position have no plane between them: the power diagram is degenerate there, and so is everything that
is computed on it ( the cells, their measures, the laplacian, the derivatives ). Two seeds `1e-9` apart have one, but the doubles
place it to `ulp( w ) / ( 2 delta )` only ( README § 23.11 ). An AGGREGATE makes both well posed, by a change of scale:

  * an aggregate `A` is a set of diracs much closer to one another than to the rest ( a SEPARATION OF SCALES, `find_groups` ).
    It has a centre `b_A` ( the barycentre of its diracs ), a scale `D_A`, and CHILDREN: seeds, or aggregates nested in it;
  * written in the frame of `A`, `y = ( x - b_A ) / D_A`, a child has the virtual position `q = ( p - b_A ) / D_A` and the
    virtual weight `v`, its true weight being `W = W_A + D_A^2 v`. The plane between two seeds of `A` is computed at their
    LOWEST COMMON AGGREGATE, in its frame: the ordinary bisector of `( q_i, V_i )` and `( q_j, V_j )` ( `V` the weight in that
    frame, the deeper levels adding `( D_child / D_A )^2` times theirs ). Every quantity is of order one there: the plane is as
    exact as the positions, at any depth ( `1e-5` then `1e-12` ), and it is the TRUE bisector ( `|x - p_i|^2 = D_A^2 |y - q_i|^2` );
  * an aggregate of EXACT duplicates ( scale 0 ) has no position to read: its children are spread on the unit sphere by
    deterministic SPRINGS ( `spring_directions` ), around an apex inside their part of the cell, at a radius `r`; their virtual
    weights split the cell of the site, which the outside sees as one seed ( their outer weights are equal );
  * seen from OUTSIDE its root aggregate, a seed is an ordinary seed: its true position and its OUTER weight `W_i` ( rounded, the
    rounding only matters where the doubles could not have placed the plane anyway ). The partial-transport ball too.

`Groups` is the tree: `find_groups( positions )` builds it from the distances alone ( `groups = "auto"`, and the solver ), nested
index lists give it explicitly ( `Groups.nested` ). The convention is that of `grill/duplicate-cut-convention/` and
`grill/near-pair-criterion/`.
"""

import numpy as np
from types import SimpleNamespace

import loom
from loom.drivers.framework_defaults import concrete


#: the separation of scales that makes an aggregate: its linkage height at most `KAPPA` times the height at which it joins the rest
KAPPA = 1e-3
#: the radius of the sphere of the virtual diracs of exact duplicates, as a fraction of the local spacing
VIRTUAL_RADIUS = 0.1
#: in an aggregate that holds other positions too, how far inside its own part the fan of a site's duplicates is centred ( a
#: fraction of the local spacing, away from the other children )
APEX_SHIFT = 0.25


class Groups:
    """THE TREE OF AGGREGATES of a diagram, in the user's order. Aggregates are numbered top-down ( a parent before its children ).

    Per seed: `parent` ( `[ n ]` ints, the aggregate the seed is a child of, `-1` when it is alone ), `dirs` ( `[ n, d ]`, the unit
    direction of a seed in an aggregate of exact duplicates, zero elsewhere ). Per aggregate: `agg_parent` ( `[ m ]`, `-1` for a root ),
    `center` ( `[ m, d ]`, the origin of its frame -- the barycentre of its diracs, or the apex of a fan of duplicates, which
    follows its site: `apex_shift`, read from `positions` ), `scale`
    ( `[ m ]`, the scale of its frame -- `r` for duplicates ), `outer` ( `[ m ]`, what its children's weights move outside: `scale`,
    or 0 for duplicates ). Optional precise virtual weights `seed_v` ( `[ n ]` ) and `agg_v` ( `[ m ]` ), each child's `v` in its
    parent's frame ( a solver's ): without them, they are read from the weights given to the diagram ( `virtual_weights` ).

    Host arrays: the tree is a constant of the positions it was built on ( a derivative goes through the positions of the
    seeds, not through the tree ). Built by `find_groups( positions )` ( `groups = "auto"`, the solver ), from nested index lists
    ( `Groups.nested`, or the lists themselves as `PowerDiagram( groups = ... )` ), or one level from representatives
    ( `Groups.from_rep` )."""

    def __init__( self, parent, agg_parent, center, scale, outer, dirs = None, seed_v = None, agg_v = None, positions = None, apex_shift = None ):
        self.parent = np.asarray( parent, dtype = np.int64 ).reshape( -1 )
        self.agg_parent = np.asarray( agg_parent, dtype = np.int64 ).reshape( -1 )
        m = len( self.agg_parent )
        self.center = np.asarray( center, dtype = np.float64 ).reshape( m, -1 )
        self.scale = np.asarray( scale, dtype = np.float64 ).reshape( -1 )
        self.outer = np.asarray( outer, dtype = np.float64 ).reshape( -1 )
        n = len( self.parent )
        d = self.center.shape[ 1 ] if m else ( np.asarray( dirs ).shape[ 1 ] if dirs is not None else 0 )
        self.dirs = np.zeros( ( n, d ) ) if dirs is None else np.asarray( dirs, dtype = np.float64 ).reshape( n, -1 )
        self.seed_v, self.agg_v = _values( seed_v ), _values( agg_v )      # ( the framework's arrays are kept: a derivative goes through )
        if ( self.agg_parent >= np.arange( m ) ).any():
            raise ValueError( "Groups: the aggregates are numbered top-down ( `agg_parent[ a ] < a`, -1 for a root )" )
        if ( self.parent >= m ).any() or ( self.parent < -1 ).any():
            raise ValueError( "Groups: `parent` gives an aggregate ( or -1 ) for each seed" )
        self._derived()
        #: per aggregate of duplicates: its apex minus the position of its site ( the fan follows the site ), zero for the others
        if apex_shift is not None:
            self.apex_shift = np.asarray( apex_shift, dtype = np.float64 ).reshape( m, -1 )
        else:
            self.apex_shift = np.zeros_like( self.center )
            dup = np.flatnonzero( self.outer == 0 )
            if len( dup ):
                if positions is None:
                    raise ValueError( "Groups: aggregates of duplicates need the positions ( or `apex_shift` )" )
                self.apex_shift[ dup ] = self.center[ dup ] - _host( positions )[ self.agg_first[ dup ] ]

    def _derived( self ):
        """the smallest seed of each aggregate, its root and its depth; the root's smallest seed and the site of each seed"""
        n, m = len( self.parent ), len( self.agg_parent )
        depth = _depths( self.agg_parent )
        first = np.full( m, n, dtype = np.int64 )
        has = self.parent >= 0
        np.minimum.at( first, self.parent[ has ], np.flatnonzero( has ) )
        for lv in range( int( depth.max() ) if m else 0, 0, -1 ):      # ( bottom-up, a level at a time )
            a = np.flatnonzero( depth == lv )
            np.minimum.at( first, self.agg_parent[ a ], first[ a ] )
        if m and ( first >= n ).any():
            raise ValueError( "Groups: an aggregate without seeds" )
        self.agg_first, self.agg_depth = first, depth
        self.agg_root = _roots( self.agg_parent )
        pa = np.maximum( self.parent, 0 )
        #: per seed: the smallest index of its root aggregate ( itself when it is alone )
        self.rep = np.arange( n, dtype = np.int64 )
        #: per seed: the smallest index at its exact position ( the first seed of its aggregate of duplicates, or itself )
        self.site = np.arange( n, dtype = np.int64 )
        if m:
            self.rep[ has ] = first[ self.agg_root[ pa[ has ] ] ]
            dup = has & ( self.outer[ pa ] == 0 )
            self.site[ dup ] = first[ pa[ dup ] ]

    # ---- what it is -----------------------------------------------------------------------------------------------

    @property
    def nb_aggregates( self ):
        return len( self.agg_parent )

    @property
    def nb_groups( self ):
        """the ROOT aggregates ( the groups of seeds that share one outer cell )"""
        return int( ( self.agg_parent < 0 ).sum() )

    @property
    def depth( self ):
        """the largest number of nested aggregates above a seed"""
        return int( self.agg_depth.max() + 1 ) if self.nb_aggregates else 0

    def members( self ):
        """`{ rep: [ members ] }` for the root aggregates ( `rep` the smallest index )"""
        res = {}
        grouped = np.flatnonzero( self.parent >= 0 )
        order = grouped[ np.lexsort( ( grouped, self.rep[ grouped ] ) ) ]
        r = self.rep[ order ]
        starts = np.flatnonzero( np.r_[ True, r[ 1: ] != r[ :-1 ] ] ) if len( r ) else np.zeros( 0, dtype = np.int64 )
        for s, e in zip( starts, np.r_[ starts[ 1: ], len( r ) ] ):
            res[ int( r[ s ] ) ] = [ int( i ) for i in order[ s:e ] ]
        return res

    def as_lists( self ):
        """the tree as nested index lists ( what `Groups.nested` takes ): one list per root aggregate"""
        kids = [ [] for _ in range( self.nb_aggregates ) ]
        for i in np.flatnonzero( self.parent >= 0 ):
            kids[ self.parent[ i ] ].append( ( int( i ), int( i ) ) )
        for a in range( self.nb_aggregates - 1, -1, -1 ):
            items = [ x for _, x in sorted( kids[ a ], key = lambda t: t[ 0 ] ) ]
            if self.agg_parent[ a ] >= 0:
                kids[ self.agg_parent[ a ] ].append( ( int( self.agg_first[ a ] ), items ) )
            else:
                kids[ a ] = items
        return [ kids[ a ] for a in sorted( np.flatnonzero( self.agg_parent < 0 ), key = lambda a: self.agg_first[ a ] ) ]

    def virtual_positions( self, positions ):
        """`[ n, d ]` host: the virtual dirac of each member of an aggregate of exact duplicates, `center + scale u` ( the points
        their split is the power diagram of ), the position of the others"""
        P = np.array( _host( positions ), dtype = np.float64 ).reshape( len( self.parent ), -1 )
        if self.nb_aggregates:
            dup = ( self.parent >= 0 ) & ( self.outer[ np.maximum( self.parent, 0 ) ] == 0 )
            a = self.parent[ dup ]
            P[ dup ] = self.center[ a ] + self.scale[ a, None ] * self.dirs[ dup ]
        return P

    def sites( self ):
        """`( keep, tree )`: the SITES ( the distinct positions: each aggregate of duplicates replaced by its first seed, which takes its
        place and its virtual weight ) and their tree ( `None` when no aggregate is left ) -- the slots of the solver are the same
        ( `x[ keep ]` )"""
        n, m = len( self.parent ), self.nb_aggregates
        dup = self.outer == 0
        keep = np.flatnonzero( self.site == np.arange( n ) )
        if not ( ~dup ).any():
            return keep, None
        new_id = np.full( m, -1, dtype = np.int64 )
        new_id[ ~dup ] = np.arange( int( ( ~dup ).sum() ) )
        pa = self.parent[ keep ]
        indup = ( pa >= 0 ) & dup[ np.maximum( pa, 0 ) ]
        pa = np.where( indup, self.agg_parent[ np.maximum( pa, 0 ) ], pa )
        parent = np.where( pa >= 0, new_id[ np.maximum( pa, 0 ) ], -1 )
        ap = self.agg_parent[ ~dup ]
        agg_parent = np.where( ap >= 0, new_id[ np.maximum( ap, 0 ) ], -1 )
        sv, av = self.seed_v, self.agg_v
        if sv is not None:
            sv_keep = sv[ keep ]
            if indup.any():                               # ( the first seed takes its aggregate's virtual weight )
                pos = np.flatnonzero( indup )
                fill = av[ self.parent[ keep[ pos ] ] ]
                sv_keep = sv_keep * _arr( sv_keep, ( ~indup ).astype( np.float64 ) ) + _scatter( fill, pos, len( keep ) )
            sv, av = sv_keep, av[ np.flatnonzero( ~dup ) ]
        res = Groups( parent, agg_parent, self.center[ ~dup ], self.scale[ ~dup ], self.outer[ ~dup ], None, sv, av,
                      apex_shift = self.apex_shift[ ~dup ] )
        res._validated = True
        return keep, res

    def with_virtual_weights( self, seed_v, agg_v ):
        """the same tree with these precise virtual weights"""
        res = Groups( self.parent, self.agg_parent, self.center, self.scale, self.outer, self.dirs, seed_v, agg_v, apex_shift = self.apex_shift )
        res._validated = getattr( self, "_validated", False )
        return res

    def __repr__( self ):
        return f"Groups( { len( self.parent ) } seeds, { self.nb_groups } groups, { self.nb_aggregates } aggregates, depth { self.depth } )"

    # ---- explicit trees ---------------------------------------------------------------------------------------------

    @staticmethod
    def nested( lists, positions, radius = None, dirs = None, apex = None ):
        """THE TREE OF NESTED INDEX LISTS: each list an aggregate, an int a seed, a list in a list a nested aggregate --
        `[ [ 0, 1 ], [ 2, [ 3, 4 ] ] ]` makes two roots, the second holding seed 2 and the aggregate `[ 3, 4 ]`. Seeds at the same
        exact position must be in one aggregate of their own ( a list of duplicates only ), or they are put in one ( the
        duplicates among an aggregate's direct seeds ). `radius` / `dirs` ( `[ n, d ]` ): the radius and the unit directions of the
        virtual diracs of duplicates, springs and `VIRTUAL_RADIUS` local spacings by default; `apex` ( `[ n, d ]`, read at each
        site's first seed ): the centres of their fans, when a point inside each site's cell is known ( the barycentre of the cell,
        which a solver has: at the weights of a transport a seed need not lie in its own cell )."""
        P = _host( positions )
        n = P.shape[ 0 ]
        parent = np.full( n, -1, dtype = np.int64 )
        agg_parent = []
        def walk( items, up ):
            a = len( agg_parent )
            agg_parent.append( up )
            for it in items:
                if isinstance( it, ( list, tuple ) ):
                    walk( it, a )
                else:
                    if parent[ int( it ) ] >= 0:
                        raise ValueError( f"Groups.nested: seed { int( it ) } is in two aggregates" )
                    parent[ int( it ) ] = a
        if lists and not isinstance( lists[ 0 ], ( list, tuple ) ):
            lists = [ lists ]
        for root in lists:
            walk( root, -1 )
        return _finish( P, parent, np.asarray( agg_parent, dtype = np.int64 ), None, radius = radius, dirs = dirs, apex = apex )

    @staticmethod
    def from_sets( sets, positions, radius = None, dirs = None, apex = None ):
        """THE TREE OF A LAMINAR FAMILY of index sets ( two sets nested or disjoint; the ones that cross another are left out ): each
        set an aggregate inside the smallest set holding it"""
        sets = sorted( { frozenset( int( i ) for i in s ) for s in sets if len( s ) > 1 }, key = lambda s: ( -len( s ), min( s ) ) )
        kept = []
        for s in sets:
            if all( s <= t or not ( s & t ) for t in kept ):
                kept.append( s )
        nodes = [ [ s, [] ] for s in kept ]
        roots = []
        for k, ( s, _ ) in enumerate( nodes ):         # ( larger first: the parent is the last larger set holding it )
            up = next( ( nodes[ j ] for j in range( k - 1, -1, -1 ) if s <= nodes[ j ][ 0 ] ), None )
            ( up[ 1 ] if up is not None else roots ).append( nodes[ k ] )
        def lists( node ):
            s, kids = node
            covered = set().union( *[ k[ 0 ] for k in kids ] ) if kids else set()
            return sorted( [ i for i in s if i not in covered ] + [ lists( k ) for k in kids ], key = lambda x: x if isinstance( x, int ) else _first( x ) )
        return Groups.nested( [ lists( r ) for r in roots ], positions, radius = radius, dirs = dirs, apex = apex )

    @staticmethod
    def from_rep( rep, positions, radius = None, dirs = None, apex = None ):
        """ONE LEVEL: `rep` ( `[ n ]` ) the smallest index of the group of each seed ( itself when alone ) -- a tree of depth one,
        the exact duplicates of a group in an aggregate of their own"""
        rep = np.asarray( rep, dtype = np.int64 ).reshape( -1 )
        lists = {}
        for i, r in enumerate( rep ):
            lists.setdefault( int( r ), [] ).append( i )
        return Groups.nested( [ v for v in lists.values() if len( v ) > 1 ], positions, radius = radius, dirs = dirs, apex = apex )


# ---- the tree from the positions ( `groups = "auto"`, the solver ) --------------------------------------------------

def find_groups( positions, *, kappa = KAPPA, duplicates = None ):
    """THE TREE OF AGGREGATES of a cloud ( host arrays ), `None` when nothing needs grouping.

    THE CRITERION is a separation of scales on the SINGLE-LINKAGE tree of the positions ( the dendrogram of their minimum spanning
    tree ): a node of linkage height `d` ( the longest edge inside it ) is an aggregate when `d <= kappa d_parent`, `d_parent` the
    height at which it joins the rest -- recursive by construction, the exact duplicates being the nodes of height 0. It reads the
    distances only ( no weight, no tolerance, no gauge ), once: an ordinary cloud has no aggregate ( its heights grow slowly ), a
    pair `1e-9` apart in it is one, and a pair `1e-12` apart inside a pair `1e-5` apart is an aggregate in an aggregate.

    THE COMPUTATION: the exact duplicates by a hash of the positions ( `duplicates`: the `( dup, first )` of
    `SdotPlanNd.exact_duplicates` when already known ), then the distinct positions' near neighbours ( a k-d tree ), whose edges are
    merged by Kruskal in BATCHES of lengths within a factor 2 ( a node and its parent in one batch are never `1 / kappa` apart ):
    each batch is one connected-components call. Vectorized, `O( n log n )`."""
    from .SdotPlanNd import exact_duplicates
    P = _host( positions )
    n, d = P.shape
    dup, first = exact_duplicates( P ) if duplicates is None else duplicates
    site = np.arange( n, dtype = np.int64 )
    site[ dup ] = first
    uniq = np.flatnonzero( site == np.arange( n ) )
    U = P[ uniq ]
    aggs, joins, nearest = _separated_nodes( U, kappa )     # ( lists of indices into `uniq`, by increasing join height )
    if not aggs and not len( dup ):
        return None
    # the aggregates of distinct positions: their parents, innermost first
    k = len( uniq )
    inner = np.full( k, -1, dtype = np.int64 )       # the innermost aggregate holding each distinct position ...
    top = np.full( k, -1, dtype = np.int64 )         # ... and the outermost one so far
    agg_parent = np.full( len( aggs ), -1, dtype = np.int64 )
    for a, mem in enumerate( aggs ):
        kids = np.unique( top[ mem ] )
        kids = kids[ kids >= 0 ]
        agg_parent[ kids ] = a
        top[ mem ] = a
        fresh = mem[ inner[ mem ] < 0 ]
        inner[ fresh ] = a
    # the seeds: each distinct position in its innermost aggregate, or in an aggregate of duplicates of its own
    u_of = np.searchsorted( uniq, site )             # the distinct position of each seed
    count = np.bincount( u_of, minlength = k )
    shared = np.flatnonzero( count > 1 )             # the distinct positions held by several seeds
    m0 = len( aggs )
    dup_agg = np.full( k, -1, dtype = np.int64 )
    dup_agg[ shared ] = m0 + np.arange( len( shared ) )
    parent = np.where( count[ u_of ] > 1, dup_agg[ u_of ], inner[ u_of ] )
    agg_parent = np.concatenate( [ agg_parent, inner[ shared ] ] )
    # the local spacing of each aggregate: the height at which its ROOT joins the rest ( the cloud's mean spacing for a lone site )
    join = np.concatenate( [ np.asarray( joins, dtype = np.float64 ), np.zeros( len( shared ) ) ] )
    h0 = _mean_spacing( U.max( axis = 0 ) - U.min( axis = 0 ), k ) if k > 1 else 1.0
    h0 = h0 if h0 > 0 else 1.0
    if len( shared ):                                # a site's own join: the distance to its nearest other position
        join[ m0: ] = nearest[ shared ] if nearest is not None else h0
    return _finish( P, parent, agg_parent, join, h0 = h0, dups_done = True )


def _finish( P, parent, agg_parent, join, radius = None, dirs = None, h0 = None, dups_done = False, apex = None ):
    """the `Groups` of a tree given by `parent` / `agg_parent` ( any numbering ): renumbered top-down, the duplicates among an
    aggregate's direct seeds put in an aggregate of their own, the centres, scales and directions computed ( vectorized )"""
    from .SdotPlanNd import exact_duplicates
    n, d = P.shape
    m = len( agg_parent )
    join = np.zeros( m ) if join is None else np.asarray( join, dtype = np.float64 )
    if h0 is None:
        h0 = _mean_spacing( P.max( axis = 0 ) - P.min( axis = 0 ), n ) if n > 1 else 1.0
        h0 = h0 if h0 > 0 else 1.0
    # the exact duplicates: one aggregate of their own, unless they are already all the seeds of theirs
    dup, first = ( np.zeros( 0, dtype = np.int64 ), ) * 2 if dups_done else exact_duplicates( P )
    if len( dup ):
        site = np.arange( n, dtype = np.int64 )
        site[ dup ] = first
        bad = parent[ dup ] != parent[ first ]
        if bad.any() or ( parent[ dup ] < 0 ).any():
            i = int( dup[ bad | ( parent[ dup ] < 0 ) ][ 0 ] )
            raise ValueError( f"groups: seed { i } is at the same position as seed { int( site[ i ] ) } but not in its aggregate -- exact "
                              "duplicates must be grouped ( `groups = 'auto'` does it )" )
        sites = np.unique( site[ dup ] )
        own = parent[ sites ]                        # the aggregate the duplicates are in
        nb_in = np.bincount( parent[ parent >= 0 ], minlength = m )
        nb_site = np.bincount( site, minlength = n )
        all_dups = nb_in[ own ] == nb_site[ sites ]
        has_kids = np.zeros( m, dtype = bool )
        has_kids[ agg_parent[ agg_parent >= 0 ] ] = True
        new = sites[ ~( all_dups & ~has_kids[ own ] ) ]  # the sites that need an aggregate of their own
        if len( new ):
            ids = m + np.arange( len( new ) )
            slot = np.full( n, -1, dtype = np.int64 )
            slot[ new ] = ids
            agg_parent = np.concatenate( [ agg_parent, parent[ new ] ] )
            join = np.concatenate( [ join, np.zeros( len( new ) ) ] )
            moved = slot[ site ] >= 0
            parent = np.where( moved, slot[ site ], parent )
            m = len( agg_parent )
    # top-down numbering: by depth
    depth = _depths( agg_parent )
    order = np.lexsort( ( np.arange( m ), depth ) )
    new_id = np.empty( m, dtype = np.int64 )
    new_id[ order ] = np.arange( m )
    agg_parent = np.where( agg_parent[ order ] >= 0, new_id[ np.maximum( agg_parent[ order ], 0 ) ], -1 )
    join = join[ order ]
    parent = np.where( parent >= 0, new_id[ np.maximum( parent, 0 ) ], -1 )
    # the barycentre and the radius of each aggregate, over all its diracs ( each seed counted in all its ancestors )
    cnt = np.zeros( m )
    acc = np.zeros( ( m, d ) )
    s = np.flatnonzero( parent >= 0 )
    a = parent[ s ].copy()
    while len( s ):
        np.add.at( cnt, a, 1.0 )
        np.add.at( acc, a, P[ s ] )
        keep = agg_parent[ a ] >= 0
        s, a = s[ keep ], agg_parent[ a[ keep ] ]
    center = acc / np.maximum( cnt, 1 )[ :, None ]
    rad = np.zeros( m )
    s = np.flatnonzero( parent >= 0 )
    a = parent[ s ].copy()
    while len( s ):
        np.maximum.at( rad, a, np.linalg.norm( P[ s ] - center[ a ], axis = 1 ) )
        keep = agg_parent[ a ] >= 0
        s, a = s[ keep ], agg_parent[ a[ keep ] ]
    # ( duplicates by the distance to the first dirac, exact: the mean of equal doubles need not be one of them )
    first = np.full( m, n, dtype = np.int64 )
    s = np.flatnonzero( parent >= 0 )
    np.minimum.at( first, parent[ s ], s )
    for lv in range( int( depth.max() ) if m else 0, 0, -1 ):
        a = np.flatnonzero( depth[ order ] == lv )
        np.minimum.at( first, agg_parent[ a ], first[ a ] )
    spread = np.zeros( m )
    s = np.flatnonzero( parent >= 0 )
    a = parent[ s ].copy()
    while len( s ):
        np.maximum.at( spread, a, np.abs( P[ s ] - P[ first[ a ] ] ).max( axis = 1 ) )
        keep = agg_parent[ a ] >= 0
        s, a = s[ keep ], agg_parent[ a[ keep ] ]
    is_dup = spread == 0
    center[ is_dup ] = P[ first[ is_dup ] ]
    outer = np.where( is_dup, 0.0, 2 * rad )         # ( the diameter bound: the scaled positions lie in a ball of radius 1 / 2 )
    has_kids = np.zeros( m, dtype = bool )
    has_kids[ agg_parent[ agg_parent >= 0 ] ] = True
    if ( is_dup & has_kids ).any():
        raise ValueError( "groups: an aggregate whose diracs are all at one position holds another aggregate ( one child only? )" )
    # the local spacing of each aggregate: its root's join height ( found ), or the cloud's mean spacing
    root = _roots( agg_parent )
    local = np.where( join[ root ] > 0, join[ root ], h0 )
    scale = outer.copy()
    scale[ is_dup ] = ( VIRTUAL_RADIUS * local[ is_dup ] ) if radius is None else radius
    # the apex of a fan of duplicates: away from the other children of its parent, into its own part
    ins = np.flatnonzero( is_dup & ( agg_parent >= 0 ) )
    if len( ins ) and radius is None:
        away = center[ ins ] - center[ agg_parent[ ins ] ]
        na = np.linalg.norm( away, axis = 1 )
        ok = na > 0
        center[ ins[ ok ] ] += APEX_SHIFT * local[ ins[ ok ], None ] * away[ ok ] / na[ ok, None ]
    if apex is not None and is_dup.any():                # ( given: read at the first seed of each aggregate of duplicates )
        center[ is_dup ] = np.asarray( apex, dtype = np.float64 ).reshape( n, d )[ first[ is_dup ] ]
    # the directions of the duplicates, by rank in their aggregate
    D = np.zeros( ( n, d ) )
    s = np.flatnonzero( ( parent >= 0 ) )
    s = s[ is_dup[ parent[ s ] ] ]
    if len( s ):
        if dirs is not None:
            D[ s ] = np.asarray( dirs, dtype = np.float64 ).reshape( n, d )[ s ]
        else:
            o = s[ np.lexsort( ( s, parent[ s ] ) ) ]
            pa = parent[ o ]
            starts = np.flatnonzero( np.r_[ True, pa[ 1: ] != pa[ :-1 ] ] )
            sizes = np.diff( np.r_[ starts, len( o ) ] )
            rank = np.arange( len( o ) ) - np.repeat( starts, sizes )
            for kk in np.unique( sizes ):            # one spring layout per size of a site
                u = spring_directions( int( kk ), d )
                sel = np.repeat( sizes, sizes ) == kk
                D[ o[ sel ] ] = u[ rank[ sel ] ]
    res = Groups( parent, agg_parent, center, scale, np.where( is_dup, 0.0, outer ), D, positions = P )
    res._validated = True                            # ( its duplicates were read on these positions )
    return res


def _separated_nodes( U, kappa ):
    """THE AGGREGATES OF DISTINCT POSITIONS `U` ( `[ k, d ]` ): `( members, joins )`, each a list of indices into `U` and the
    height at which it joins its parent, by increasing join -- the nodes of the single-linkage tree with `d <= kappa d_parent` --
    and the distance from each position to its nearest other one"""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    k, d = U.shape
    if k < 3:
        return [], [], None
    K = min( k - 1, 2 * d + 1 )                  # ( the minimum spanning tree is in the near neighbours, Delaunay's degree ~ 2 d )
    tree = cKDTree( U )
    dist, idx = tree.query( U, k = K + 1, workers = -1 )
    a = np.repeat( np.arange( k, dtype = np.int64 ), K )
    b = idx[ :, 1: ].reshape( -1 ).astype( np.int64 )
    l = dist[ :, 1: ].reshape( -1 )
    a, b, l = _connect( U, tree, a, b, l, K )
    # ONLY THE SHORT EDGES: an aggregate's height is at most `kappa` times its join, itself at most the longest edge -- the
    # components of the edges up to `kappa max l` are the only nodes that can be aggregates ( their inner nodes included )
    every = ( a, b, l )
    short = l <= kappa * l.max()
    a, b, l = a[ short ], b[ short ], l[ short ]
    # Kruskal by batches: the edges of lengths in [ 2^e, 2^( e + 1 ) )
    ex = np.frexp( l )[ 1 ].astype( np.int16 )
    order = np.argsort( ex, kind = "stable" )    # ( a radix sort on small ints )
    a, b, l, ex = a[ order ], b[ order ], l[ order ], ex[ order ]
    starts = np.flatnonzero( np.r_[ True, ex[ 1: ] != ex[ :-1 ] ] )
    ends = np.r_[ starts[ 1: ], len( ex ) ]
    label = np.arange( k, dtype = np.int64 )     # the component of each position ( its smallest index )
    height = np.zeros( k )                       # per component label: the longest edge inside ...
    size = np.ones( k, dtype = np.int64 )        # ... and its number of positions
    mark = np.zeros( k, dtype = bool )
    slot = np.zeros( k, dtype = np.int64 )
    members, joins = [], []
    for s, e in zip( starts, ends ):
        la, lb = label[ a[ s:e ] ], label[ b[ s:e ] ]
        cross = la != lb
        if not cross.any():
            continue
        la, lb, ll = la[ cross ], lb[ cross ], l[ s:e ][ cross ]
        # the components this batch merges: each joins at its shortest edge here
        mark[ la ] = True                        # ( the labels involved, without a sort )
        mark[ lb ] = True
        inv_l = np.flatnonzero( mark )
        mark[ inv_l ] = False
        slot[ inv_l ] = np.arange( len( inv_l ) )
        inv = slot[ np.r_[ la, lb ] ]
        jn = np.full( len( inv_l ), np.inf )
        np.minimum.at( jn, inv, np.r_[ ll, ll ] )
        sep = ( size[ inv_l ] > 1 ) & ( height[ inv_l ] <= kappa * jn )
        for c, j in zip( inv_l[ sep ], jn[ sep ] ):
            members.append( np.flatnonzero( label == c ) )
            joins.append( float( j ) )
        # merge ( on the labels involved only )
        nl = len( inv_l )
        g = coo_matrix( ( np.ones( len( la ) ), ( inv[ :len( la ) ], inv[ len( la ): ] ) ), shape = ( nl, nl ) )
        nc, comp = connected_components( g, directed = False )
        new = np.full( nc, k, dtype = np.int64 )
        np.minimum.at( new, comp, inv_l )
        hgt = np.zeros( nc )
        sz = np.zeros( nc, dtype = np.int64 )
        np.maximum.at( hgt, comp, height[ inv_l ] )
        np.add.at( sz, comp, size[ inv_l ] )
        np.maximum.at( hgt, comp[ inv[ :len( la ) ] ], ll )
        remap = np.arange( k, dtype = np.int64 )
        remap[ inv_l ] = new[ comp ]
        label = remap[ label ]
        height[ new ] = hgt
        size[ new ] = sz
    # the components left, joined by the long edges: each one's join is its shortest edge out
    la, lb = label[ every[ 0 ] ], label[ every[ 1 ] ]
    cross = la != lb
    jn = np.full( k, np.inf )
    np.minimum.at( jn, la[ cross ], every[ 2 ][ cross ] )
    np.minimum.at( jn, lb[ cross ], every[ 2 ][ cross ] )
    tops = np.flatnonzero( ( label == np.arange( k ) ) & ( size > 1 ) & ( height <= kappa * jn ) & np.isfinite( jn ) )
    if len( tops ):
        order = np.argsort( label, kind = "stable" )
        starts = np.searchsorted( label[ order ], tops )
        for c, s0 in zip( tops, starts ):
            members.append( order[ s0:s0 + size[ c ] ] )
            joins.append( float( jn[ c ] ) )
    order = np.argsort( np.asarray( joins ), kind = "stable" )
    return [ members[ i ] for i in order ], [ joins[ i ] for i in order ], dist[ :, 1 ]


def _connect( U, tree, a, b, l, K ):
    """the near-neighbour edges, plus what joins their components when they are several ( a cluster of more than `K` close points
    away from the rest ): for each component but the largest, its shortest edge out found among more neighbours ( Boruvka )"""
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components
    k = U.shape[ 0 ]
    for rnd in range( 64 ):
        if rnd == 0:                                 # ( the near neighbours, `K` per row in order: a CSR as it is, no sort )
            g = csr_matrix( ( np.ones( len( a ), dtype = np.int8 ), b, np.arange( 0, len( a ) + 1, K ) ), shape = ( k, k ) )
        else:
            g = coo_matrix( ( np.ones( len( a ), dtype = np.int8 ), ( a, b ) ), shape = ( k, k ) )
        nc, comp = connected_components( g, directed = True, connection = "weak" )
        if nc <= 1:
            break
        big = np.bincount( comp ).argmax()
        pts = np.flatnonzero( comp != big )
        kk = min( k - 1, 4 * K )
        best = {}
        while True:
            dist, idx = tree.query( U[ pts ], k = kk + 1, workers = -1 )
            other = comp[ idx ] != comp[ pts ][ :, None ]
            other[ :, 0 ] = False
            hit = other.any( axis = 1 )
            col = other.argmax( axis = 1 )
            for p, h, c in zip( pts, hit, col ):
                if h:
                    dd = dist[ pts.searchsorted( p ), c ]
                    cp = comp[ p ]
                    if cp not in best or dd < best[ cp ][ 2 ]:
                        best[ cp ] = ( p, idx[ pts.searchsorted( p ), c ], dd )
            if len( best ) == nc - 1 or kk >= k - 1:
                break
            kk = min( k - 1, 4 * kk )
        if not best:
            break
        e = np.asarray( list( best.values() ) )
        a = np.r_[ a, e[ :, 0 ].astype( np.int64 ) ]
        b = np.r_[ b, e[ :, 1 ].astype( np.int64 ) ]
        l = np.r_[ l, e[ :, 2 ] ]
    return a, b, l


def _depths( agg_parent ):
    """the depth of each aggregate ( 0 for a root ), vectorized"""
    m = len( agg_parent )
    depth = np.zeros( m, dtype = np.int64 )
    up = agg_parent.copy()
    for _ in range( m ):
        live = up >= 0
        if not live.any():
            break
        depth[ live ] += 1
        up[ live ] = agg_parent[ up[ live ] ]
    return depth


def _roots( agg_parent ):
    """the root of each aggregate, vectorized"""
    root = np.arange( len( agg_parent ), dtype = np.int64 )
    for _ in range( len( agg_parent ) ):
        up = agg_parent[ root ]
        if not ( up >= 0 ).any():
            break
        root = np.where( up >= 0, up, root )
    return root


def _mean_spacing( ext, n ):
    """the mean spacing of `n` points spread over a box of extents `ext` ( the flat directions left out )"""
    e = ext[ ext > 0 ]
    if len( e ) == 0:
        return 0.0
    return float( np.prod( e ) / n ) ** ( 1.0 / len( e ) )


def spring_directions( k, d ):
    """`k` unit DIRECTIONS in `d` dimensions, the same for the same `( k, d )`: the virtual diracs of `k` exact duplicates, by rank.
    1D: the ranks in order ( `-1 .. 1` ); 2D: the regular polygon, the first at angle 0; beyond: a deterministic start relaxed by
    SPRINGS ( a repulsion between every two of them, on the unit sphere ) -- what matters is that they are spread and deterministic,
    not their precision."""
    if k == 1:
        return np.zeros( ( 1, d ) )
    if d == 1:
        return np.linspace( -1.0, 1.0, k )[ :, None ]
    if d == 2:
        t = 2 * np.pi * np.arange( k ) / k
        return np.stack( [ np.cos( t ), np.sin( t ) ], axis = 1 )
    rng = np.random.default_rng( 1234 + 7919 * k + d )
    u = rng.normal( size = ( k, d ) )
    u /= np.linalg.norm( u, axis = 1, keepdims = True )
    for _ in range( 200 ):
        diff = u[ :, None, : ] - u[ None, :, : ]
        dist = np.linalg.norm( diff, axis = 2 ) + np.eye( k )
        force = ( diff / dist[ :, :, None ] ** 3 ).sum( axis = 1 )
        u = u + 0.05 * force / k
        u /= np.linalg.norm( u, axis = 1, keepdims = True )
    return u


# ---- what the storages receive ---------------------------------------------------------------------------------

def as_groups( groups, positions ):
    """a `Groups` from what `PowerDiagram( groups = ... )` takes: a `Groups`, or nested index lists"""
    if isinstance( groups, Groups ):
        return groups
    if isinstance( groups, ( list, tuple ) ):
        if loom.is_traced( getattr( positions, "raw", positions ) ):
            raise ValueError( "groups: nested lists need readable positions ( their frames are read from them ) -- build a `Groups` outside the trace" )
        return Groups.nested( list( groups ), positions )
    raise TypeError( f"groups: a `Groups`, nested index lists, `'auto'` or `None` ( got { type( groups ).__name__ } )" )


def virtual_weights( groups, weights ):
    """`( seed_v, agg_v )`: each child's virtual weight in its parent's frame, read from the weights `w` given seed by seed
    ( `v = ( w[ first seed of the child ] - w[ first seed of the parent ] ) / scale_parent^2`; the roots 0 ) -- framework gathers, a
    derivative goes back through them. The duplicates' weights given are their virtual weights ( `w = W_site + r^2 v` )."""
    g = groups
    n = len( g.parent )
    w = getattr( weights, "raw", weights )
    has = np.flatnonzero( g.parent >= 0 )
    pa = g.parent[ has ]
    s_idx = np.zeros( n, dtype = np.int64 )
    s_idx[ has ] = g.agg_first[ pa ]
    inv2 = np.zeros( n )
    inv2[ has ] = 1.0 / g.scale[ pa ] ** 2
    seed_v = ( w - w[ s_idx ] ) * _arr( w, inv2 )
    m = g.nb_aggregates
    kid = np.flatnonzero( g.agg_parent >= 0 )
    a_idx = np.array( g.agg_first, dtype = np.int64 )
    up = np.array( g.agg_first, dtype = np.int64 )
    up[ kid ] = g.agg_first[ g.agg_parent[ kid ] ]
    ainv2 = np.zeros( m )
    ainv2[ kid ] = 1.0 / g.scale[ g.agg_parent[ kid ] ] ** 2
    agg_v = ( w[ a_idx ] - w[ up ] ) * _arr( w, ainv2 ) if m else w[ :0 ]
    return seed_v, agg_v


def resolve( groups, positions, weights ):
    """THE ARRAYS OF A GROUPED DIAGRAM, in the user's order: `( outer_weights, members )` -- the outer weight of each seed ( that of
    its site: `w[ site ]` ), and the dict of `group_rep`, `group_next`, `group_parent` ( host ints ), `group_dirs`, `group_agg_*`
    ( host ), `group_vweights` / `group_agg_vweights` ( the precise ones of the tree, or the framework's arrays derived from the
    weights by gathers ). `weights` may be `None` ( zeros )."""
    pos = getattr( positions, "raw", positions )
    n = int( pos.shape[ 0 ] )
    g = groups
    if g.parent.shape != ( n, ):
        raise ValueError( f"groups: a tree of { len( g.parent ) } seeds for { n } positions" )
    if not getattr( g, "_validated", False ) and not loom.is_traced( pos ):    # ( duplicates outside one aggregate of theirs )
        from .SdotPlanNd import exact_duplicates
        dup, first = exact_duplicates( _host( pos ) )
        bad = ( g.site[ dup ] != g.site[ first ] ) if len( dup ) else np.zeros( 0, dtype = bool )
        if bad.any():
            i = int( dup[ bad ][ 0 ] )
            raise ValueError( f"groups: seed { i } is at the same position as seed { int( first[ bad ][ 0 ] ) } but not in its aggregate of "
                              "duplicates -- exact duplicates must be grouped ( `groups = 'auto'` does it )" )
    w = getattr( weights, "raw", weights )
    if w is None:
        w = pos[ :, 0 ] * 0                              # ( zeros, in the framework and the float of the positions )
    outer = w[ g.site ]
    if g.seed_v is not None:
        seed_v, agg_v = g.seed_v, g.agg_v
    else:
        seed_v, agg_v = virtual_weights( g, w )
    m = g.nb_aggregates
    d = int( pos.shape[ 1 ] )
    return outer, dict( group_rep = g.rep.astype( np.int32 ), group_next = next_members( g.rep ).astype( np.int32 ),
                        group_parent = g.parent.astype( np.int32 ), group_vweights = seed_v, group_dirs = g.dirs.reshape( n, d ),
                        group_agg_parent = g.agg_parent.astype( np.int32 ),
                        group_agg_center = np.where( g.outer[ :, None ] == 0, g.apex_shift, g.center ).reshape( m, d ),
                        group_agg_scale = g.scale, group_agg_outer = g.outer, group_agg_vweights = agg_v )


def next_members( rep ):
    """the next member of each seed's group, in increasing index, the last one going back to the first ( a cycle; itself when
    alone ) -- vectorized"""
    n = len( rep )
    order = np.lexsort( ( np.arange( n ), rep ) )
    r = rep[ order ]
    nxt_sorted = np.r_[ order[ 1: ], order[ :1 ] ]
    last = np.r_[ r[ 1: ] != r[ :-1 ], True ]                # the last of its group in the sorted order
    starts = np.flatnonzero( np.r_[ True, r[ 1: ] != r[ :-1 ] ] )
    sizes = np.diff( np.r_[ starts, n ] )
    first_of = np.repeat( order[ starts ], sizes )
    nxt = np.empty( n, dtype = np.int64 )
    nxt[ order ] = np.where( last, first_of, nxt_sorted )
    return nxt


def _host( x ):
    from loom.tensor.host import to_host
    a = np.asarray( to_host( getattr( x, "raw", x ) ), dtype = np.float64 )
    return a.reshape( a.shape[ 0 ], -1 ) if a.ndim != 2 else a


# ---- the inner levels of Newton, frozen ( `SdotPlanNd._split_groups`, `_attach_split_derivatives` ) ------------------------

def _tree_maps( g ):
    """the maps of a tree that `TreeSystem` reads ( host ): `par`, `first`, `slot`, `expand`, `common`, `kids`, `child_of`,
    `subtree`, `below`, `anc` -- a node is a seed ( `< n` ) or `n + a` for the aggregate `a`"""
    n, m = len( g.parent ), g.nb_aggregates
    par = lambda node: g.parent[ node ] if node < n else g.agg_parent[ node - n ]
    first = lambda node: node if node < n else int( g.agg_first[ node - n ] )
    scale = lambda a: 1.0 if a < 0 else g.scale[ a ]

    def slot( node ):                                # the slot of a node's value, or -1 ( the first child: the gauge )
        p = par( node )
        return first( node ) if p < 0 or first( node ) != g.agg_first[ p ] else -1

    def expand( i, L ):                              # the weight of seed `i` in the frame of `L`, as `[ ( slot, coef ) ]`
        chain, node = [ i ], i
        while par( node ) != L:
            node = n + par( node )
            chain.append( node )
        res, f = [], 1.0
        for k in range( len( chain ) - 1, -1, -1 ):
            s = slot( chain[ k ] )
            if s >= 0 and f != 0:
                res.append( ( s, f ) )
            if k and chain[ k ] >= n:                # ( entering this aggregate: its children's weights scaled by `( o / s_parent )^2` )
                a = chain[ k ] - n
                f = f * ( g.outer[ a ] / scale( g.agg_parent[ a ] ) ) ** 2
        return res

    def ancestors( i ):
        res, a = [], g.parent[ i ]
        while a >= 0:
            res.append( a )
            a = g.agg_parent[ a ]
        return res

    anc = { int( i ): ancestors( int( i ) ) for i in np.flatnonzero( g.parent >= 0 ) }

    def common( i, j ):
        if g.rep[ i ] != g.rep[ j ]:
            return -1
        cj = set( anc[ j ] )
        return next( a for a in anc[ i ] if a in cj )

    # the children of each aggregate, and the child of `a` that holds seed `i`
    kids = [ [] for _ in range( m ) ]
    for i in np.flatnonzero( g.parent >= 0 ):
        kids[ g.parent[ i ] ].append( int( i ) )
    for a in range( m ):
        if g.agg_parent[ a ] >= 0:
            kids[ g.agg_parent[ a ] ].append( n + a )
    def child_of( i, a ):
        node = i
        while par( node ) != a:
            node = n + par( node )
        return node
    subtree = [ { a } for a in range( m ) ]           # the aggregates under each one, itself included
    for a in range( m - 1, -1, -1 ):
        if g.agg_parent[ a ] >= 0:
            subtree[ g.agg_parent[ a ] ] |= subtree[ a ]
    below = [ set() for _ in range( m ) ]
    for i, an in anc.items():
        for a in an:
            below[ a ].add( i )

    return SimpleNamespace( n = n, m = m, par = par, first = first, scale = scale, slot = slot, expand = expand, anc = anc,
                            common = common, kids = kids, child_of = child_of, subtree = subtree, below = below )


class TreeSystem:
    """THE INNER LEVELS OF NEWTON on a tree of aggregates, from the facets of a diagram ( frozen ): `sdotplan/Groups.h::direction`
    without the root level, written as gathers so that it applies to host arrays as to the framework's ( a derivative goes through
    it: the implicit function theorem of `_attach_split_derivatives` ).

    The UNKNOWNS are those of the solver, one SLOT per seed ( `sdotplan/Groups.h` ): a root's outer weight ( a lone seed is its own
    root ), or the virtual weight of a child that is not the first of its aggregate. `step( b, D )` completes a direction `D` whose
    root slots are set ( `PowerDiagram.laplacian_solve`, or zeros ): aggregate by aggregate, a level at a time from the roots down,
    the sums of the masses of each child ( but the first ) answer `b`, the facets out of it counted at the slots already known --
    one sweep of block Gauss-Seidel, exact but for the coupling between aggregates of one level and back to the levels above
    ( `( o / s )^2`, of order `kappa^2` ).

    `rows`, `counts`, `ids`, `vals`: `PowerDiagram.hessian_rows( items = rows )` of a diagram with these groups ( each coefficient
    with respect to the weights in the frame of the lowest common aggregate of its two seeds ), `rows` holding every member."""

    def __init__( self, g, rows, counts, ids, vals ):
        self.g = g
        n, m = len( g.parent ), g.nb_aggregates
        self.n = n
        t = _tree_maps( g )
        par, first, slot, expand, anc, common = t.par, t.first, t.slot, t.expand, t.anc, t.common
        kids, child_of, subtree, below = t.kids, t.child_of, t.subtree, t.below

        # the facets of each member
        facets = { int( i ): [] for i in rows }
        for r, i in enumerate( rows ):
            for q in range( int( counts[ r ] ) ):
                j, c = int( ids[ r, q ] ), float( vals[ r, q ] )
                if j >= 0 and j != i and c != 0:
                    facets[ int( i ) ].append( ( j, c ) )

        # per aggregate, its block: the rows ( children but the first ), `B`, the seeds summed, the known slots counted
        blocks = []
        for a in range( m ):
            unk = [ k for k in kids[ a ] if first( k ) != g.agg_first[ a ] ]
            if not unk:
                continue
            rank = { first( k ): u for u, k in enumerate( unk ) }
            inner = below[ a ] - { int( g.agg_first[ a ] ) }       # the slots below `a`: unknown at its level ( its own children's ) or later
            B = np.zeros( ( len( unk ), len( unk ) ) )
            seeds = [ [] for _ in unk ]
            known = [ {} for _ in unk ]
            for i in below[ a ]:
                u = rank.get( first( child_of( i, a ) ), -1 )
                if u < 0:
                    continue
                seeds[ u ].append( i )
                cu = child_of( i, a )
                inside = subtree[ cu - n ] if cu >= n else ()
                for j, c in facets.get( i, [] ):
                    L = common( i, j ) if j in anc else -1
                    if L in inside:
                        continue                         # inside the child: cancels in its sum
                    terms = {}
                    for s, f in expand( i, L ):
                        terms[ s ] = terms.get( s, 0.0 ) + c * f
                    for s, f in expand( j, L ):
                        terms[ s ] = terms.get( s, 0.0 ) - c * f
                    for s, v in terms.items():
                        if s in rank:
                            B[ u, rank[ s ] ] += v
                        elif s not in inner:
                            known[ u ][ s ] = known[ u ].get( s, 0.0 ) - v
            try:
                Binv = np.linalg.inv( B )
            except np.linalg.LinAlgError:
                Binv = np.linalg.pinv( B )
            blocks.append( ( int( g.agg_depth[ a ] ), [ first( k ) for k in unk ], seeds, known, Binv ) )

        # by level: the gathers of each one
        self.levels = []
        for lv in sorted( { b[ 0 ] for b in blocks } ):
            bl = [ b for b in blocks if b[ 0 ] == lv ]
            out = [ s for b in bl for s in b[ 1 ] ]                # the slots this level sets, one row each
            sum_idx = _pad( [ sd for b in bl for sd in b[ 2 ] ], n )
            kn = [ kd for b in bl for kd in b[ 3 ] ]
            kn_idx = _pad( [ list( kd.keys() ) for kd in kn ], n )
            kn_coef = _pad( [ list( kd.values() ) for kd in kn ], 0.0, dtype = np.float64 )
            inv_idx, inv_coef, at = [], [], 0              # the block inverses, as gathers over the level's rows
            for b in bl:
                k = len( b[ 1 ] )
                for u in range( k ):
                    inv_idx.append( list( range( at, at + k ) ) )
                    inv_coef.append( list( b[ 4 ][ u ] ) )
                at += k
            self.levels.append( ( np.asarray( out, dtype = np.int64 ), sum_idx, kn_idx, kn_coef,
                                  _pad( inv_idx, len( out ) ), _pad( inv_coef, 0.0, dtype = np.float64 ) ) )

        self._maps( t )

    def _maps( self, t ):
        """the outer weight of each member as slots ( `W_i = sum coef D[ slot ]` ), and the masks of the virtual weights"""
        g, n, m = self.g, t.n, t.m
        expand, kids, first = t.expand, t.kids, t.first
        mem = np.flatnonzero( g.parent >= 0 )
        ex = [ expand( int( i ), -1 ) for i in mem ]
        self.members = mem
        self.outer_idx = _pad( [ [ s for s, _ in e ] for e in ex ], n )
        self.outer_coef = _pad( [ [ f for _, f in e ] for e in ex ], 0.0, dtype = np.float64 )
        self.root_slot = np.ones( n, dtype = bool )
        for a in range( m ):
            for k in kids[ a ]:
                if first( k ) != g.agg_first[ a ]:
                    self.root_slot[ first( k ) ] = False
        self.seed_mask = np.zeros( n )
        self.seed_mask[ mem ] = [ 0.0 if g.agg_first[ g.parent[ i ] ] == i else 1.0 for i in mem ]
        self.agg_mask = np.array( [ 0.0 if g.agg_parent[ a ] < 0 or g.agg_first[ a ] == g.agg_first[ g.agg_parent[ a ] ] else 1.0 for a in range( m ) ] )

    @classmethod
    def from_rows( cls, g, rows, counts, ids, vals ):
        """THE SAME SYSTEM from the backend's rows ( `PowerDiagram.hessian_rows( items = rows, host = False )` ), under a trace too
        ( `jax.jit` ). Its STRUCTURE comes from the tree alone: the blocks, their rows, the seeds each row sums, and the row that each
        facet of a member feeds ( the aggregate and its child that hold the member, at each level ). What a facet gives is GATHERED
        from the traced rows: the lowest common aggregate of its two seeds ( their chains of ancestors compared ), the expansions
        of their weights in its frame, the column of the block ( a one-hot over its children ) or the known slot each term reads.
        The blocks are inverted by Gauss-Jordan on the backend ( they are positive definite ), all of it frozen ( `stop_gradient` ).
        `rows` must be the members of `g`, in order ( `self.members` )."""
        self = cls.__new__( cls )
        self.g = g
        t = _tree_maps( g )
        n, m = t.n, t.m
        self.n = n
        self._maps( t )
        mem = self.members
        if not np.array_equal( np.asarray( rows, dtype = np.int64 ).reshape( -1 ), mem ):
            raise ValueError( "TreeSystem.from_rows: the rows must be the members of the tree, in order" )
        self.levels, self.flevels = [], []

        # the blocks ( as in `__init__` ): per aggregate with unknowns, its rows ( children but the first ) and their seeds
        blocks, rank_of = [], {}
        for a in range( m ):
            unk = [ k for k in t.kids[ a ] if t.first( k ) != g.agg_first[ a ] ]
            if not unk:
                continue
            rk = { t.first( k ): u for u, k in enumerate( unk ) }
            seeds = [ [] for _ in unk ]
            for i in sorted( t.below[ a ] ):
                u = rk.get( t.first( t.child_of( i, a ) ), -1 )
                if u >= 0:
                    seeds[ u ].append( i )
            blocks.append( ( int( g.agg_depth[ a ] ), a, [ t.first( k ) for k in unk ], seeds ) )
            rank_of[ a ] = rk
        if not blocks:
            return self
        K = max( len( b[ 2 ] ) for b in blocks )

        # THE TABLES, per seed ( a last row `n` for the padding ): its chain of ancestors ( `CH[ s, l ]`, then `-1` for the roots'
        # level ), the expansion of its weight in the frame of each ( `EXS`, `EXF` ), the aggregate whose child owns its slot and
        # that child's row ( `OWNP`, `RANK` )
        R = len( mem )
        dep = np.zeros( n + 1, dtype = np.int64 )
        for i, c in t.anc.items():
            dep[ i ] = len( c )
        Dm = int( dep.max() )
        W = Dm + 1
        CH = np.full( ( n + 1, W ), -2, dtype = np.int64 )
        CH[ :, 0 ] = -1
        ANC = np.full( ( n + 1, Dm ), -3, dtype = np.int64 )
        EXS = np.full( ( n + 1, W, W ), n, dtype = np.int64 )
        EXF = np.zeros( ( n + 1, W, W ) )
        EXS[ :n, 0, 0 ] = np.arange( n )
        EXF[ :n, 0, 0 ] = 1.0
        OWNP = np.full( n + 1, -3, dtype = np.int64 )
        RANK = np.full( n + 1, -1, dtype = np.int64 )
        for i, c in t.anc.items():
            CH[ i, :len( c ) ] = c
            CH[ i, len( c ) ] = -1
            ANC[ i, :len( c ) ] = c
            EXS[ i ], EXF[ i ] = n, 0.0
            for l in range( len( c ) + 1 ):
                for q, ( sl, f ) in enumerate( t.expand( i, c[ l ] if l < len( c ) else -1 ) ):
                    EXS[ i, l, q ], EXF[ i, l, q ] = sl, f
            node = i
            while t.par( node ) >= 0 and g.agg_first[ t.par( node ) ] == i:
                node = n + t.par( node )
            P = t.par( node )
            OWNP[ i ] = P
            if P >= 0:
                RANK[ i ] = rank_of[ P ][ i ]
        # per member and level: the row of the block of its ancestor that it feeds ( -1: the first child, none )
        U = np.full( ( R, Dm ), -1, dtype = np.int64 )
        for r, i in enumerate( mem ):
            for l, a in enumerate( t.anc[ int( i ) ] ):
                if a in rank_of:
                    U[ r, l ] = rank_of[ a ].get( t.first( t.child_of( int( i ), a ) ), -1 )

        # THE FACETS ( traced ): `j` the neighbour ( `n` when the entry is not a facet ), `lam` the level of the lowest common
        # aggregate in the chain of the member ( its depth when there is none: `L = -1` ), `lamj` in the chain of `j`
        ops = loom.ops( vals )
        # ( the tables as the backend's arrays: under a `jit`, constants of the program, which a traced index can read )
        cst = lambda a: a if isinstance( vals, np.ndarray ) else ops.concatenate( [ a ], axis = 0 )
        I = lambda a: cst( _arr( vals, np.asarray( a, dtype = np.int64 ), concrete( integer = True ) ) )
        F = lambda a: cst( _arr( vals, np.asarray( a, dtype = np.float64 ) ) )
        vals = ops.stop_gradient( vals )
        Q = int( ids.shape[ 1 ] )
        valid = ( I( np.arange( Q ) )[ None, : ] < counts.reshape( -1 )[ :, None ] ) & ( ids >= 0 ) & ( ids != I( mem )[ :, None ] ) & ( vals != 0 )
        j = ops.where( valid, ids, n )
        ANCt, DEPt = I( ANC ), I( dep )
        Aj = ANCt[ j ]                                                     # [ R, Q, Dm ]
        match = ops.any( I( CH[ mem, :Dm ] )[ :, None, :, None ] == Aj[ :, :, None, : ], axis = 3 )
        before = ops.cumsum( match * 1, axis = 2 ) == 0
        lam = ops.sum( before * I( ( np.arange( Dm )[ None, : ] < dep[ mem ][ :, None ] ) * 1 )[ :, None, : ], axis = 2 )
        L = I( CH.reshape( -1 ) )[ I( mem * W )[ :, None ] + lam ]
        lamj = ops.where( L < 0, DEPt[ j ], ops.sum( ( Aj == L[ :, :, None ] ) * I( np.arange( Dm ) ), axis = 2 ) )
        EXSt, EXFt = I( EXS.reshape( -1, W ) ), F( EXF.reshape( -1, W ) )
        at_i, at_j = I( mem * W )[ :, None ] + lam, j * W + lamj
        S = ops.concatenate( [ EXSt[ at_i ], EXSt[ at_j ] ], axis = 2 )   # [ R, Q, 2W ]: the slots of the terms
        V = ops.concatenate( [ EXFt[ at_i ], -EXFt[ at_j ] ], axis = 2 ) * vals[ :, :, None ]
        OS, RS, AS = I( OWNP )[ S ], I( RANK )[ S ], ANCt[ S ]

        # per level of the member's chain, the rows that feed a block: their row of `B` ( one-hot ) and their known terms
        brows, kcoef, kslot, dest = [], [], [], []
        for l in range( Dm ):
            rr = np.flatnonzero( ( l < dep[ mem ] ) & ( U[ :, l ] >= 0 ) )
            if not len( rr ):
                continue
            a = CH[ mem[ rr ], l ]
            ri, aa = I( rr ), I( a )[ :, None, None ]
            on = ( valid[ ri ] & ( lam[ ri ] >= l ) )[ :, :, None ] * 1.0
            inr = ( OS[ ri ] == aa ) & ( RS[ ri ] >= 0 )
            inn = ops.any( AS[ ri ] == aa[ :, :, :, None ], axis = 3 ) & ( S[ ri ] != I( g.agg_first[ a ] )[ :, None, None ] )
            v = V[ ri ] * on
            hot = ( RS[ ri ][ :, :, :, None ] == I( np.arange( K ) ) ) * ( v * inr )[ :, :, :, None ]
            brows.append( ops.sum( ops.reshape( hot, ( len( rr ), -1, K ) ), axis = 1 ) )
            kn = ( 1 - inr * 1 ) * ( 1 - inn * 1 )
            kcoef.append( ops.reshape( -v * kn, ( len( rr ), -1 ) ) )
            kslot.append( ops.reshape( S[ ri ], ( len( rr ), -1 ) ) )
            dest += [ ( int( a_ ), int( U[ r, l ] ) ) for r, a_ in zip( rr, a ) ]
        E = len( dest )
        Ball = ops.concatenate( brows + [ brows[ 0 ][ :1 ] * 0 ], axis = 0 )   # ( a zero row last: the padding )
        self.kcoef = ops.concatenate( kcoef, axis = 0 )
        self.kslot = ops.concatenate( kslot, axis = 0 )

        # by level: the rows, the entries summed into each, the blocks inverted
        for lv in sorted( { b[ 0 ] for b in blocks } ):
            bl = [ b for b in blocks if b[ 0 ] == lv ]
            out = [ sl for b in bl for sl in b[ 2 ] ]
            row_of = {}
            for b in bl:
                for u in range( len( b[ 2 ] ) ):
                    row_of[ ( b[ 1 ], u ) ] = len( row_of )
            ent = [ [] for _ in out ]
            for e, d in enumerate( dest ):
                if d in row_of:
                    ent[ row_of[ d ] ].append( e )
            ent_idx = _pad( ent, E )
            nr = len( out )
            blk = np.full( ( len( bl ), K ), nr, dtype = np.int64 )            # the rows of each block, padded
            pad = np.zeros( ( len( bl ), K ) )
            for bi, b in enumerate( bl ):
                k = len( b[ 2 ] )
                blk[ bi, :k ] = [ row_of[ ( b[ 1 ], u ) ] for u in range( k ) ]
                pad[ bi, k: ] = 1.0
            Brow = _gather_rows( Ball, ent_idx )                               # [ nr, K ]
            Bb = ops.concatenate( [ Brow, Brow[ :1 ] * 0 ], axis = 0 )[ I( blk ) ] + F( pad[ :, :, None ] * np.eye( K )[ None ] )
            unblock = np.concatenate( [ bi * K + np.arange( len( b[ 2 ] ) ) for bi, b in enumerate( bl ) ] )
            self.flevels.append( ( np.asarray( out, dtype = np.int64 ), _pad( [ sd for b in bl for sd in b[ 3 ] ], n ), ent_idx, blk, unblock,
                                   _inverse( Bb ) ) )
        return self

    def _step_rows( self, b, D ):
        """`step` for a system made by `from_rows`"""
        ops = loom.ops( D )
        for out, sum_idx, ent_idx, blk, unblock, Binv in self.flevels:
            De = ops.concatenate( [ D, D[ :1 ] * 0 ], axis = 0 )
            kv = ops.sum( self.kcoef * De[ self.kslot ], axis = 1 )
            rhs = _gather_sum( b, sum_idx, None ) + _gather_sum( kv, ent_idx, None )
            rb = ops.concatenate( [ rhs, rhs[ :1 ] * 0 ], axis = 0 )[ blk ]
            d = ops.reshape( ops.matmul( Binv, rb[ :, :, None ] ), ( -1, ) )[ unblock ]
            D = D + _scatter( d, out, self.n )
        return D

    def step( self, b, D ):
        """`D` completed level by level ( see the class ): `b` the right-hand side per seed ( `nu - m` ), `D` the slots with the roots'
        ones set ( others zero )"""
        if getattr( self, "flevels", None ):
            return self._step_rows( b, D )
        for out, sum_idx, kn_idx, kn_coef, inv_idx, inv_coef in self.levels:
            rhs = _gather_sum( b, sum_idx, None ) + _gather_sum( D, kn_idx, kn_coef )
            d = _gather_sum( rhs, inv_idx, inv_coef )
            D = D + _scatter( d, out, self.n )
        return D

    def outer( self, x ):
        """the outer weight of each seed for the slots `x`"""
        mem = self.members
        vals = _gather_sum( x, self.outer_idx, self.outer_coef )
        keep = np.ones( self.n )
        keep[ mem ] = 0.0
        return x * _arr( x, keep ) + _scatter( vals, mem, self.n )

    def virtual_weights( self, x ):
        """`( seed_v, agg_v )` for the slots `x`"""
        g = self.g
        return x * _arr( x, self.seed_mask ), x[ g.agg_first ] * _arr( x, self.agg_mask )

    def weights( self, x ):
        """`groups.py`'s weights for the slots `x`: the outer ones, plus `s^2 v` for a duplicate"""
        g = self.g
        extra = np.zeros( self.n )
        dup = np.flatnonzero( ( g.parent >= 0 ) )
        dup = dup[ g.outer[ g.parent[ dup ] ] == 0 ]
        extra[ dup ] = g.scale[ g.parent[ dup ] ] ** 2 * self.seed_mask[ dup ]
        return self.outer( x ) + x * _arr( x, extra )

    def slot_sum( self, r, rows = None ):
        """`R[ s ] = sum_k coef_ks r_k`: the right-hand side `r` ( per seed, or per seed of `rows` ) summed onto the slots through the
        outer weights ( `W_k = sum_s coef_ks x_s` ) -- so that `sum_k W_k r_k = sum_s x_s R_s` is written with the precise slots,
        each root's residuals summed first"""
        rows = np.arange( self.n ) if rows is None else np.asarray( rows, dtype = np.int64 )
        at = np.full( self.n, -1, dtype = np.int64 )
        at[ self.members ] = np.arange( len( self.members ) )
        lists = [ [] for _ in range( self.n ) ]
        for k, i in enumerate( rows ):
            if at[ i ] < 0:
                lists[ i ].append( ( k, 1.0 ) )
            else:
                for s, f in zip( self.outer_idx[ at[ i ] ], self.outer_coef[ at[ i ] ] ):
                    if s < self.n and f != 0:
                        lists[ s ].append( ( k, f ) )
        idx = _pad( [ [ k for k, _ in l ] for l in lists ], len( rows ) )
        coef = _pad( [ [ f for _, f in l ] for l in lists ], 0.0, dtype = np.float64 )
        return _gather_sum( r, idx, coef )

    def slots_of( self, w, seed_v = None, agg_v = None ):
        """the slots for weights `w` given seed by seed: the roots' outer weights from `w`, the virtual weights from `seed_v` /
        `agg_v` ( relative to the first child of each aggregate: gathers, the backend's arrays too ) or read from `w` ( host,
        `virtual_weights` )"""
        g, n = self.g, self.n
        if seed_v is None:
            w = np.asarray( w, dtype = np.float64 ).reshape( -1 )
            seed_v, agg_v = virtual_weights( g, w )
            seed_v, agg_v = np.asarray( seed_v ), np.asarray( agg_v )
        w, seed_v, agg_v = ( _values( v ) for v in ( w, seed_v, agg_v ) )
        tgt, src, sub = [], [], []
        for i in np.flatnonzero( ( g.parent >= 0 ) & ( self.seed_mask > 0 ) ):
            tgt.append( i ); src.append( i ); sub.append( _first_kid_at( g, g.parent[ i ], n ) )
        for a in np.flatnonzero( self.agg_mask > 0 ):
            tgt.append( g.agg_first[ a ] ); src.append( n + a ); sub.append( _first_kid_at( g, g.agg_parent[ a ], n ) )
        x = w.reshape( -1 ) * _arr( w, self.root_slot * 1.0 )
        if not tgt:
            return x
        ops = loom.ops( w ) if not isinstance( w, np.ndarray ) else None
        parts = [ v.reshape( -1 ) if ops is None or not isinstance( v, np.ndarray ) else _arr( w, v.reshape( -1 ) ) for v in ( seed_v, agg_v ) ]
        z = np.concatenate( parts ) if ops is None else ops.concatenate( parts, axis = 0 )   # ( host pieces in w's framework: Torch's `cat` takes no numpy )
        src, sub = np.asarray( src, dtype = np.int64 ), np.asarray( sub, dtype = np.int64 )
        return x + _scatter( z[ src ] - z[ sub ], np.asarray( tgt, dtype = np.int64 ), n )


def _first_kid_at( g, a, n ):
    """where the virtual weight of the first child of aggregate `a` is, in `concatenate( [ seed_v, agg_v ] )`"""
    f = g.agg_first[ a ]
    if g.parent[ f ] == a:
        return int( f )
    k = np.flatnonzero( ( g.agg_parent == a ) & ( g.agg_first == f ) )
    return n + int( k[ 0 ] )


def _first( lists ):
    """the smallest index in nested lists"""
    return min( x if isinstance( x, int ) else _first( x ) for x in lists )


def _values( v ):
    """host values as float arrays, the framework's as they are"""
    if v is None or not isinstance( v, ( np.ndarray, list, tuple ) ):
        return v
    return np.asarray( v, dtype = np.float64 ).reshape( -1 )


def _pad( rows, fill, dtype = np.int64 ):
    """`[ r, width ]`: the lists padded with `fill`"""
    width = max( [ len( r ) for r in rows ] + [ 1 ] )
    res = np.full( ( len( rows ), width ), fill, dtype = dtype )
    for k, r in enumerate( rows ):
        res[ k, :len( r ) ] = r
    return res


def _arr( x, a, dtype = None ):
    """`a` ( host ) in the framework, on the device and in the float of `x` -- or in `dtype` ( a loom `Dtype` ): x's, not the
    default framework's ( a Jax value under a Torch default stays Jax )"""
    if isinstance( x, np.ndarray ):
        return a if dtype is None else np.asarray( a, dtype = dtype.numpy_dtype )
    from loom.tensor.storage.buffers import buffer_class_for
    from loom.devices.Device import Device
    ops, where = loom.ops( x ), buffer_class_for( x )( x ).device()        # `( "cuda", 0 )`, None for a traced value
    device = None if where is None else Device.factory( "cpu" if where[ 0 ] == "cpu" else f"cuda:{ where[ 1 ] }" )
    return ops.array( a, ops.dtype_of( x ) if dtype is None else dtype, device )


def _gather_sum( x, idx, coef ):
    """`sum_t x[ idx[ :, t ] ] coef[ :, t ]`, an index past the end reading zero"""
    xe = np.concatenate( [ x, np.zeros( 1 ) ] ) if isinstance( x, np.ndarray ) else loom.ops( x ).concatenate( [ x, x[ :1 ] * 0 ], axis = 0 )
    res = None
    for t in range( idx.shape[ 1 ] ):
        v = xe[ idx[ :, t ] ]
        if coef is not None:
            v = v * _arr( x, coef[ :, t ] )
        res = v if res is None else res + v
    return res


def _scatter( vals, where, n ):
    """a vector of `n`, `vals` at `where`, zero elsewhere ( a gather: any framework )"""
    pos = np.full( n, len( where ), dtype = np.int64 )
    pos[ where ] = np.arange( len( where ) )
    ve = np.concatenate( [ vals, np.zeros( 1 ) ] ) if isinstance( vals, np.ndarray ) else loom.ops( vals ).concatenate( [ vals, vals[ :1 ] * 0 ], axis = 0 )
    return ve[ pos ]


def _gather_rows( X, idx ):
    """`sum_t X[ idx[ :, t ] ]` over the rows of `X`, whose last row is the padding ( zero )"""
    res = None
    for t in range( idx.shape[ 1 ] ):
        v = X[ idx[ :, t ] ]
        res = v if res is None else res + v
    return res


def _inverse( B ):
    """the inverses of a stack of positive definite blocks `[ nb, k, k ]`: numpy's on host arrays, Gauss-Jordan without pivots on
    the backend's ( a trace: `k` steps of whole-stack operations )"""
    if isinstance( B, np.ndarray ):
        try:
            return np.linalg.inv( B )
        except np.linalg.LinAlgError:
            return np.linalg.pinv( B )
    ops = loom.ops( B )
    k = int( B.shape[ 1 ] )
    M, X = B, B * 0 + _arr( B, np.eye( k ) )[ None ]
    for p in range( k ):
        piv = M[ :, p, p ][ :, None ]
        piv = ops.where( piv != 0, piv, piv * 0 + 1 )    # ( a block of an empty cell: finite, as `pinv` )
        rm, rx = M[ :, p, : ] / piv, X[ :, p, : ] / piv
        col = M[ :, :, p ][ :, :, None ]
        at = _arr( B, ( np.arange( k ) == p ) * 1.0 )[ None, :, None ]
        M = ( M - col * rm[ :, None, : ] ) * ( 1 - at ) + at * rm[ :, None, : ]
        X = ( X - col * rx[ :, None, : ] ) * ( 1 - at ) + at * rx[ :, None, : ]
    return X
