from errand import test


# GROUPS OF SEEDS ( `PowerDiagram( groups = ... )`, `sdot/groups.py` ): a tree of aggregates, seeds that share one cell and split it
# among themselves -- each plane computed in the frame of the lowest common aggregate of its two seeds, at the scale of that
# aggregate; exact duplicates through their virtual diracs. Checked on the degenerate clouds of `test_SdotPlanNd.py::_degenerate_clouds`
# ( the same moves, and a pair at `1e-12` inside a pair at `1e-5` ), with the double kernel.


def _degenerate_clouds( d, seed ):
    """`{ name: ( positions, groups ) }`: a uniform cloud with seeds moved onto others ( each move relative to the previous
    member, so that a move nests ); `groups` the expected clusters ( `( members, exact )`, `exact`: equal positions ) -- as in
    `test_SdotPlanNd.py`"""
    rng = numpy.random.default_rng( seed )
    n = 400 if d == 2 else 600
    base = rng.uniform( 0.05, 0.95, size = ( n, d ) )
    unit = numpy.ones( d ) / d ** 0.5
    side = numpy.eye( d )[ 0 ]
    def moved( *moves ):
        pos = base.copy()
        groups = []
        for members, offsets in moves:
            for m, prev, off in zip( members[ 1: ], members[ :-1 ], offsets ):
                pos[ m ] = ( pos[ members[ 0 ] ] if off is not None and off.ndim == 1 else pos[ prev ] ) + ( off if off.ndim == 1 else off[ 0 ] )
            groups.append( ( members, all( not numpy.any( off ) for off in offsets ) ) )
            if any( off.ndim == 2 for off in offsets ):              # ( the inner pair is a cluster of its own, nested )
                groups.append( ( members[ 1: ], False ) )
        return pos, groups
    return {
        "exact pair":         moved( ( [ 5, 7 ], [ 0.0 * unit ] ) ),
        "exact triple":       moved( ( [ 3, 11, 29 ], [ 0.0 * unit, 0.0 * unit ] ) ),
        "pair at 1e-9":       moved( ( [ 5, 7 ], [ 1e-9 * unit ] ) ),
        "pair at 1e-12":      moved( ( [ 8, 2 ], [ 1e-12 * unit ] ) ),
        "triple at 1e-10":    moved( ( [ 4, 9, 13 ], [ 1e-10 * unit, 2e-10 * side ] ) ),
        "pair and duplicate": moved( ( [ 1, 6 ], [ 3e-10 * unit ] ), ( [ 12, 15, 30 ], [ 0.0 * unit, 2e-10 * side ] ) ),
        # ( a row `[ off ]`: from the PREVIOUS member -- a pair `1e-12` apart, `1e-5` from the first )
        "nested 1e-5 / 1e-12": moved( ( [ 20, 21, 22 ], [ 1e-5 * unit, 1e-12 * side[ None ] ] ) ),
    }


def _sites( pos ):
    """the smallest index at the same exact position"""
    from sdot.SdotPlanNd import exact_duplicates
    site = numpy.arange( len( pos ) )
    dup, first = exact_duplicates( pos )
    site[ dup ] = first
    return site


if test( "auto_groups_the_seeds_the_doubles_cannot_separate" ):
    import numpy
    from loom.testing import need
    from sdot import PowerDiagram, box_half_spaces
    need( "cpu" )
    # `groups = "auto"` finds every expected cluster ( exact or past the floor of the doubles ) -- and maybe a few ordinary close pairs
    # ( a group is never wrong, its planes are the true ones ). The members TILE the cell of their group -- for a site of exact duplicates, the cell of one seed there ( the diagram without the duplicates ) --
    # every cell is finite and the measures sum to the domain
    for d in ( 2, 3 ):
        bnd = box_half_spaces( [ 0.0 ] * d, [ 1.0 ] * d )
        for name, ( pos, expected ) in _degenerate_clouds( d, 17 ).items():
            n = len( pos )
            pd = PowerDiagram( pos, numpy.zeros( n ), boundaries = bnd, groups = "auto", kernel_dtype = "FP64" )
            found = pd.groups.members()
            rep = numpy.asarray( pd.groups.rep )
            for members, _ in expected:
                assert ( rep[ members ] == rep[ members[ 0 ] ] ).all(), ( d, name, members, found )
            extra = [ g for g in found.values() if not any( set( m ) <= set( g ) for m, _ in expected ) ]
            for g in extra:                              # an extra group: seeds much closer than the spacing ( a long facet counts more )
                assert max( numpy.linalg.norm( pos[ a ] - pos[ b ] ) for a in g for b in g ) < 0.5 * ( 1 / n ) ** ( 1 / d ), ( d, name, g )
            m = numpy.asarray( pd.measures ).reshape( -1 )
            assert numpy.isfinite( m ).all() and ( m >= 0 ).all() and abs( m.sum() - 1 ) < 1e-12, ( d, name, m.sum() )
            site = _sites( pos )
            keep = numpy.flatnonzero( site == numpy.arange( n ) )
            ref = numpy.asarray( PowerDiagram( pos[ keep ], numpy.zeros( len( keep ) ), boundaries = bnd, kernel_dtype = "FP64" ).measures ).reshape( -1 )
            for k, s in enumerate( keep ):
                members = numpy.flatnonzero( site == s )
                assert abs( m[ members ].sum() - ref[ k ] ) < 1e-14, ( d, name, s, m[ members ].sum() - ref[ k ] )
                if len( members ) > 1:                   # each duplicate has a piece of it
                    assert ( m[ members ] > 1e-3 * ref[ k ] ).all(), ( d, name, m[ members ] )
            print( f"  { d }D { name }: { found }" )

    # a cloud without close seeds has no group ( a jittered grid )
    rng = numpy.random.default_rng( 3 )
    pos = ( numpy.stack( numpy.meshgrid( numpy.arange( 17 ), numpy.arange( 17 ) ), axis = -1 ).reshape( -1, 2 ) + 0.5 + rng.uniform( -0.3, 0.3, size = ( 289, 2 ) ) ) / 17
    assert PowerDiagram( pos, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), groups = "auto" ).groups is None


if test( "the_kernel_cuts_the_groups_as_the_oracle" ):
    import numpy
    from loom.testing import need
    from sdot import PowerDiagram, box_half_spaces
    need( "cpu" )
    # the same planes as the Python oracle ( `PowerDiagram.cell( i )`, one call per cut ), on both storages, in 2D and 3D: a site of
    # four duplicates with weights that differ ( they move its virtual split ), a pair, a close pair grouped, an aggregate nested in
    # an aggregate with a duplicate in it
    from sdot.groups import Groups, find_groups
    rng = numpy.random.default_rng( 1 )
    for d in ( 2, 3 ):
        n = 30
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        pos[ 1 ] = pos[ 0 ]
        pos[ [ 7, 9, 11 ] ] = pos[ 4 ]
        pos[ 13 ] = pos[ 12 ] + 1e-4
        pos[ 16 ] = pos[ 15 ] + 3e-3
        pos[ 17 ] = pos[ 16 ] + 2e-6
        pos[ 18 ] = pos[ 17 ]
        w = rng.uniform( -0.01, 0.01, size = n )
        w[ 16 ] = w[ 15 ] + 1e-4                         # ( the inner weights at the scale of their aggregates )
        w[ 17 ] = w[ 16 ] + 1e-10
        g = Groups.nested( [ [ 0, 1 ], [ 4, 7, 9, 11 ], [ 12, 13 ], [ 15, [ 16, 17, 18 ] ] ], pos )
        assert g.depth == 3, g
        bnd = box_half_spaces( [ 0.0 ] * d, [ 1.0 ] * d )
        for acc in ( None, "plain" ):
            pd = PowerDiagram( pos, w, boundaries = bnd, groups = g, accelerator = acc, kernel_dtype = "FP64" )
            m = numpy.asarray( pd.measures ).reshape( -1 )
            oracle = numpy.array( [ float( numpy.asarray( pd.cell( i ).measure ).reshape( -1 )[ 0 ] ) for i in range( n ) ] )
            assert numpy.abs( m - oracle ).max() < 1e-14, ( d, acc, numpy.abs( m - oracle ).max() )
            # the cells drawn are the same as the measured ones
            cells = pd.cells
            vol = numpy.asarray( cells.measure ).reshape( -1 )
            assert numpy.abs( vol - m ).max() < 1e-14, ( d, acc )


if test( "a_close_pair_tends_to_the_split_of_its_virtual_diracs" ):
    import numpy
    from loom.testing import need
    from sdot import PowerDiagram, box_half_spaces
    need( "cpu" )
    # THE CONVENTION IS A LIMIT: two diracs at `p +- delta u` ( an ordinary diagram ) tend, as `delta -> 0`, to the exact duplicates
    # whose virtual diracs are `p +- r u` ( any `r`: the split does not depend on it ) -- and, with weights, to virtual weights
    # scaled by `r / delta`
    from sdot.groups import Groups
    rng = numpy.random.default_rng( 5 )
    for d in ( 2, 3 ):
        n = 20
        base = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        u = rng.normal( size = d )
        u /= numpy.linalg.norm( u )
        bnd = box_half_spaces( [ 0.0 ] * d, [ 1.0 ] * d )
        gap = 0.003                                      # a weight gap of the pair, at the scale of the virtual diracs
        for r in ( 0.01, 0.2 ):
            pos = base.copy()
            pos[ 1 ] = pos[ 0 ]
            dirs = numpy.zeros( ( n, d ) )
            dirs[ 0 ], dirs[ 1 ] = - u, u
            w = numpy.zeros( n )
            w[ 1 ] = gap
            g = Groups.nested( [ [ 0, 1 ] ], pos, radius = r, dirs = dirs )
            grouped = numpy.asarray( PowerDiagram( pos, w, boundaries = bnd, groups = g, kernel_dtype = "FP64" ).measures ).reshape( -1 )
            for delta in ( 1e-5, 1e-7 ):
                near = base.copy()
                near[ 0 ] -= delta * u
                near[ 1 ] = base[ 0 ] + delta * u
                wn = numpy.zeros( n )
                wn[ 1 ] = gap * delta / r
                m = numpy.asarray( PowerDiagram( near, wn, boundaries = bnd, kernel_dtype = "FP64" ).measures ).reshape( -1 )
                err = numpy.abs( m - grouped ).max()
                assert err < 20 * delta, ( d, r, delta, err )
                print( f"  { d }D r = { r }, delta = { delta :g}: { err :.1e}" )


if test( "groups_derive_like_seeds" ):
    import numpy
    from loom.testing import check_grad, need, need_autodiff
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.distributions.Image import Image
    from sdot.groups import Groups, find_groups
    need( "cpu" )
    need_autodiff()
    # the measures and the moments of a grouped diagram against the positions ( of the SITES: the members of a site of duplicates
    # move together, where the cells are differentiable ), the weights ( the outer ones and the virtual ones of the duplicates )
    # and the density; a close pair is grouped at `1e-3` ( a finite difference must stay far below its gap ), and nested
    rng = numpy.random.default_rng( 0 )
    for d in ( 2, 3 ):
        n = 12
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        pos[ 1 ] = pos[ 0 ]
        pos[ [ 7, 9 ] ] = pos[ 4 ]
        pos[ 3 ] = pos[ 2 ] + 1e-3
        pos[ 5 ] = pos[ 3 ] + 3e-2
        w = rng.uniform( -0.01, 0.01, size = n )
        g = Groups.nested( [ [ 0, 1 ], [ 4, 7, 9 ], [ 5, [ 2, 3 ] ] ], pos )
        sites = numpy.unique( g.site )
        expand = lambda ps: ps[ numpy.searchsorted( sites, g.site ) ]
        bnd = box_half_spaces( [ 0.0 ] * d, [ 1.0 ] * d )
        tight = dict( eps = 1e-6, rtol = 1e-6, atol = 1e-9 )
        dists = [ None ] if d == 3 else [ None, Image( rng.uniform( 0.5, 1.5, size = ( 3, 3 ) ), frame = [ 1.0, 1.0 ] ) ]
        for dist in dists:
            for k in ( None, *range( d + 2 ) ):
                def f( ps, q ):
                    pd = PowerDiagram( expand( ps ), q, boundaries = bnd, groups = g, kernel_dtype = "FP64", distribution = dist )
                    return pd.measures if k is None else pd.moment( k )
                check_grad( f, pos[ sites ], w, seed = 3 + ( 0 if k is None else k ), **tight )
        # the tree, the weights traced alone
        tree = AaBsp( pos, w )
        f = lambda q: PowerDiagram( pos, q, boundaries = bnd, groups = g, kernel_dtype = "FP64", accelerator = tree ).measures
        check_grad( f, w, seed = 7, **tight )

    # ( calls jax itself: under another framework, only the generic part above runs )
    from loom.drivers import framework_defaults
    if str( framework_defaults.framework() ) == "jax":
        # ONE duplicate moved alone is not differentiable ( the cells around see its site through planes that coincide ): its share is
        # the mean of the one-sided derivatives -- each member of the site the same column, the sum that of the site moved as a whole
        import jax, jax.numpy as jnp
        n = 12
        pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
        pos[ 1 ] = pos[ 0 ]
        w = rng.uniform( -0.01, 0.01, size = n )
        g = find_groups( pos )
        bnd = box_half_spaces( [ 0.0 ] * 2, [ 1.0 ] * 2 )
        meas = lambda p: ( lambda m: jnp.asarray( getattr( m, "raw", m ) ) )( PowerDiagram( p, w, boundaries = bnd, groups = g, kernel_dtype = "FP64" ).measures )
        J = numpy.asarray( jax.jacrev( meas )( jnp.asarray( pos ) ) )
        rows = [ i for i in range( n ) if i not in ( 0, 1 ) ]             # the cells around ( the members see their own virtual planes )
        assert numpy.abs( J[ rows, 0 ] - J[ rows, 1 ] ).max() < 1e-12
        e = numpy.zeros( ( n, 2 ) )
        e[ [ 0, 1 ], 0 ] = 1e-7
        fd = ( numpy.asarray( meas( jnp.asarray( pos + e ) ) ) - numpy.asarray( meas( jnp.asarray( pos - e ) ) ) ) / 2e-7
        assert numpy.abs( fd - J[ :, 0, 0 ] - J[ :, 1, 0 ] ).max() < 1e-6, numpy.abs( fd - J[ :, 0, 0 ] - J[ :, 1, 0 ] ).max()


if test( "a_sink_keeps_the_balls_on_the_true_positions" ):
    import numpy
    from loom.testing import need
    from sdot import PowerDiagram, box_half_spaces
    need( "cpu" )
    # partial transport: the members of a site integrate in the ball of their site ( the virtual diracs only split the cell ), so they
    # tile what one seed there would get
    rng = numpy.random.default_rng( 4 )
    n = 40
    pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
    pos[ 1 ] = pos[ 0 ]
    pos[ [ 7, 9, 11 ] ] = pos[ 4 ]
    w = 0.01 + rng.uniform( -1e-4, 1e-4, size = n )     # ( close to equal: each seed lies in its cell, each member's ball meets its piece )
    w[ 1 ] = w[ 0 ]
    w[ [ 7, 9, 11 ] ] = w[ 4 ]
    bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    pd = PowerDiagram( pos, w, boundaries = bnd, groups = "auto", kernel_dtype = "FP64", sink_weight = 0.0 )
    m = numpy.asarray( pd.measures ).reshape( -1 )
    keep = [ i for i in range( n ) if i not in ( 1, 7, 9, 11 ) ]
    ref = numpy.asarray( PowerDiagram( pos[ keep ], w[ keep ], boundaries = bnd, kernel_dtype = "FP64", sink_weight = 0.0 ).measures ).reshape( -1 )
    assert abs( m[ [ 0, 1 ] ].sum() - ref[ 0 ] ) < 1e-14 and abs( m[ [ 4, 7, 9, 11 ] ].sum() - ref[ keep.index( 4 ) ] ) < 1e-14
    assert ( m[ [ 0, 1, 4, 7, 9, 11 ] ] > 0 ).all(), m


if test( "groups_run_under_jit_and_refuse_what_they_cannot_read" ):
    import numpy
    import pytest
    from loom.testing import need, need_autodiff
    from sdot import PowerDiagram, box_half_spaces
    from sdot.groups import find_groups
    need( "cpu" )
    need( "jax" )                                   # ( the test calls jax itself )
    import jax, jax.numpy as jnp
    rng = numpy.random.default_rng( 2 )
    n = 20
    pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
    pos[ 1 ] = pos[ 0 ]
    w = numpy.zeros( n )
    bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    g = find_groups( pos )
    eager = numpy.asarray( PowerDiagram( pos, w, boundaries = bnd, groups = g, kernel_dtype = "FP64" ).measures ).reshape( -1 )
    jitted = jax.jit( lambda p: ( lambda m: getattr( m, "raw", m ) )( PowerDiagram( p, w, boundaries = bnd, groups = g, kernel_dtype = "FP64" ).measures ) )( jnp.asarray( pos ) )
    assert numpy.abs( numpy.asarray( jitted ).reshape( -1 ) - eager ).max() < 1e-14
    # `"auto"` reads the positions: not under a trace
    with pytest.raises( ValueError, match = "readable" ):
        jax.jit( lambda p: PowerDiagram( p, w, boundaries = bnd, groups = "auto" ).measures.raw )( jnp.asarray( pos ) )
    # not with a periodicity yet
    with pytest.raises( NotImplementedError, match = "periodicity" ):
        PowerDiagram( pos, w, boundaries = bnd, groups = g, periodicity = True )
    # an exact duplicate outside the group of its position
    from sdot.groups import Groups
    with pytest.raises( ValueError, match = "same position" ):
        PowerDiagram( pos, w, boundaries = bnd, groups = [ [ 0, 2 ] ] )


if test( "the_frozen_levels_read_the_backend_rows" ):
    import numpy
    from loom.testing import need, need_autodiff
    from sdot import PowerDiagram, box_half_spaces
    from sdot.groups import TreeSystem, find_groups
    need( "cpu" )
    need( "jax" )                                   # ( the test calls jax itself )
    import jax, jax.numpy as jnp
    # `TreeSystem.from_rows` ( the backend's rows, gathers: what a `jit` takes ) is the system of the host's rows, its step the
    # same to the rounding -- eagerly and inside a `jit`, on every degenerate cloud ( the nested one couples its levels )
    bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    rng = numpy.random.default_rng( 4 )
    for name, ( pos, _ ) in _degenerate_clouds( 2, 31 ).items():
        n = len( pos )
        g = find_groups( pos )
        mem = numpy.flatnonzero( g.parent >= 0 )
        w = 1e-4 * rng.normal( size = n )
        w[ mem ] = w[ g.rep[ mem ] ]                     # ( the outer weights of a root's members are one: its own )
        pd = lambda w: PowerDiagram( pos, w, boundaries = bnd, groups = g, kernel_dtype = "FP64" )
        ref = TreeSystem( g, mem, *pd( w ).hessian_rows( items = mem ) )
        b = rng.normal( size = n )
        D = rng.normal( size = n ) * ref.root_slot
        want = ref.step( b, D )
        def step( w, b, D ):
            ts = TreeSystem.from_rows( g, mem, *pd( w ).hessian_rows( items = mem, host = False ) )
            return ts.step( b, D )
        args = [ jnp.asarray( v ) for v in ( w, b, D ) ]
        for got in ( step( *args ), jax.jit( step )( *args ) ):
            err = numpy.abs( numpy.asarray( got ) - want ).max()
            assert err < 1e-10 * numpy.abs( want ).max(), ( name, err )
        # the slots of given weights, the same through gathers
        sv, av = numpy.zeros( n ), numpy.zeros( g.nb_aggregates )
        sv[ mem ], av[ : ] = rng.normal( size = len( mem ) ), rng.normal( size = g.nb_aggregates )
        x = ref.slots_of( w, sv, av )
        assert numpy.abs( numpy.asarray( jax.jit( lambda w, s, a: ref.slots_of( w, s, a ) )( *[ jnp.asarray( v ) for v in ( w, sv, av ) ] ) ) - x ).max() < 1e-15, name
