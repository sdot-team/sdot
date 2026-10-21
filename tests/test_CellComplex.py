from errand import test, skip


# THE CELL COMPLEX ( `CellComplex.py` ): the cells of a diagram glued along their faces. What is checked here does not go
# through the gluing: the EULER CHARACTERISTIC of the complex ( 1 for a subdivision of a convex domain, `( -1 )^d` for one of the
# whole space ), the measures against those of the cells ( `Cell.measure` ) and of the facets against the hessian of the
# measures ( `hessian_rows`: `int_facet 1 / ( 2 | p_i - p_j | )` without weights ), and the orientation of the normals against
# the seeds -- on random seeds, on a GRID ( the degenerate case: 2^d cells at a vertex ), and in FP32.


def _np( x ):
    """a number of the complex ( a tensor of the default framework ) as numpy"""
    import numpy
    from loom.tensor.host import to_host
    return numpy.asarray( to_host( x ) )


def _box( d ):
    import numpy
    from sdot.PowerDiagram import box_half_spaces
    return box_half_spaces( numpy.zeros( d ), numpy.ones( d ) )


def _check( pd, euler = 1, periodic = False ):
    """the invariants of a complex, which do not depend on how it was glued"""
    import numpy
    cx = pd.cell_complex()
    d = cx.dim
    # the Euler characteristic, over the cells that are not empty
    nb = [ cx.nb_faces( k ) for k in range( d ) ] + [ int( ( numpy.diff( cx.cells.incident( 0 )[ 0 ] ) > 0 ).sum() ) ]
    chi = sum( ( -1 ) ** k * nb[ k ] for k in range( d + 1 ) )
    assert chi == euler, ( chi, nb )
    # a facet between two seeds is in their two cells ( one when a cell meets an image of itself ), a facet on the domain in one
    sides = cx.facets.sides
    off, _ = cx.facets.incident( d )
    two = ( sides[ :, 1 ] >= 0 ) & ( sides[ :, 0 ] != sides[ :, 1 ] )
    assert ( ( numpy.diff( off ) == 2 ) == two ).all()
    assert ( ( numpy.diff( off ) == 1 ) == ~two ).all()
    if not periodic:
        # the normal goes from the first seed to the second ( the bisector of a power diagram is orthogonal to `p_j - p_i` )
        pos = numpy.asarray( pd.positions ).reshape( -1, d )
        inner = sides[ :, 1 ] >= 0
        dp = pos[ sides[ inner, 1 ] ] - pos[ sides[ inner, 0 ] ]
        cos = ( dp * _np( cx.facets.normals )[ inner ] ).sum( axis = 1 ) / numpy.linalg.norm( dp, axis = 1 )
        assert numpy.allclose( cos, 1, atol = 1e-5 ), cos.min()
    # the measures of the cells, by the pyramids over their faces from vertices computed from their generators, against the cells
    # cut in doubles ( those of an FP32 diagram are less precise than the complex )
    bounded = cx.cells.is_bounded
    ref = numpy.asarray( pd._folded_cells( "FP64" ).measure ).reshape( -1 )
    assert numpy.allclose( numpy.asarray( _np( cx.cells.measures ) )[ bounded ], ref[ bounded ], atol = 1e-10 )
    # children and parents are the two sides of one relation
    for k in range( 1, d + 1 ):
        co, ci = cx.incident( k, k - 1 )
        po, pi = cx.incident( k - 1, k )
        down = { ( f, int( g ) ) for f in range( len( co ) - 1 ) for g in ci[ co[ f ] : co[ f + 1 ] ] }
        up = { ( int( f ), g ) for g in range( len( po ) - 1 ) for f in pi[ po[ g ] : po[ g + 1 ] ] }
        assert down == up, k
    return cx


if test( "random_seeds" ):
    import numpy
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 0 )
    for d in ( 2, 3 ):
        p = rng.random( ( 40, d ) )
        cx = _check( PowerDiagram( p, rng.random( 40 ) * 0.01, boundaries = _box( d ) ) )
        inner = cx.facets.sides[ :, 1 ] >= 0
        s = cx.facets.sides[ inner ]
        assert numpy.allclose( _np( cx.facets.seed_vectors )[ inner ], p[ s[ :, 1 ] ] - p[ s[ :, 0 ] ] )

        # the area of each facet, against the hessian of the measures ( no weights: `vals = area / ( 2 | p_i - p_j | )` )
        pd = PowerDiagram( p, boundaries = _box( d ) )
        cx = _check( pd )
        cnt, ids, vals = pd.hessian_rows()
        for f, ( a, b ) in enumerate( cx.facets.sides ):
            if b < 0:
                continue
            r = list( ids[ a, : cnt[ a ] ] ).index( b )
            assert abs( vals[ a, r ] * 2 * numpy.linalg.norm( p[ a ] - p[ b ] ) - _np( cx.facets.measures )[ f ] ) < 1e-9

        # the walls: a facet on the domain lies on the half-space its side names, and its normal is the outward one
        dirs, offs = cx.walls
        for f in numpy.flatnonzero( cx.facets.sides[ :, 1 ] < 0 ):
            w = -1 - cx.facets.sides[ f, 1 ]
            assert numpy.allclose( _np( cx.faces( d - 1 )[ int( f ) ].points ) @ dirs[ w ], offs[ w ] )
            assert numpy.allclose( _np( cx.facets.normals )[ f ], dirs[ w ] )


if test( "grid" ):
    # the degenerate case: 2^d cells around each inner vertex, which the generators of a vertex do not name alone
    import numpy
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    for kd in ( "FP64", "FP32" ):
        set_kernel_dtype( kd )
        for d in ( 2, 3 ):
            m = 4
            g = ( numpy.stack( numpy.meshgrid( *[ numpy.arange( m ) + 0.5 ] * d, indexing = "ij" ), -1 ).reshape( -1, d ) ) / m
            cx = _check( PowerDiagram( g, boundaries = _box( d ) ) )
            # the faces of a grid of `m^d` cubes: `C( d, k ) m^( d - k ) ( m + 1 )^k` k-faces... summed over the directions
            from math import comb
            for k in range( d + 1 ):
                expected = comb( d, k ) * m ** k * ( m + 1 ) ** ( d - k )
                assert cx.nb_faces( k ) == expected, ( kd, d, k, cx.nb_faces( k ), expected )
            assert numpy.allclose( _np( cx.faces( 1 ).measures ), 1 / m, atol = 1e-6 )
    set_kernel_dtype( "FP64" )


if test( "unbounded" ):
    # without a domain: the faces that run off to infinity are there, unbounded, and the complex is one of the whole space
    import numpy
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 1 )
    for d in ( 2, 3 ):
        cx = _check( PowerDiagram( rng.random( ( 12, d ) ) ), euler = ( -1 ) ** d )
        assert not cx.cells.is_bounded.all() and cx.cells.is_bounded.any()
        assert numpy.isinf( _np( cx.cells.measures )[ ~cx.cells.is_bounded ] ).all()


if test( "polygon_domain" ):
    # a domain that is not a box ( the support of a `ConvexHull` ): the walls are its facets
    import numpy
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    from sdot.distributions.Polytope import ConvexHull
    set_kernel_dtype( "FP64" )
    tri = numpy.array( [ [ 0, 0 ], [ 1, 0 ], [ 0.5, 1.0 ] ] )
    rng = numpy.random.default_rng( 2 )
    pd = PowerDiagram( rng.random( ( 30, 2 ) ) * [ 1, 0.5 ] + [ 0, 0.1 ], distribution = ConvexHull( tri ) )
    cx = _check( pd )
    on = cx.facets.sides[ :, 1 ] < 0
    assert abs( _np( cx.facets.measures )[ on ].sum() - ( 1 + 2 * numpy.hypot( 0.5, 1 ) ) ) < 1e-9     # the perimeter
    assert abs( _np( cx.cells.measures ).sum() - 0.5 ) < 1e-9


if test( "walk" ):
    # the `Face` views: what a user walks with
    import numpy
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 3 )
    cx = PowerDiagram( rng.random( ( 25, 3 ) ), boundaries = _box( 3 ) ).cell_complex()
    for f in cx.facets:
        assert f.dim == 2 and len( f.cells ) in ( 1, 2 )
        assert { c.seed for c in f.cells } == { s for s in f.sides if s >= 0 }
        for e in f.children:
            assert f in e.parents
            assert set( e.vertices ) <= set( f.vertices )
        # the vertices of a facet in the order of its boundary, counterclockwise around its normal
        P = _np( f.points )
        area = 0.5 * numpy.cross( P - P.mean( axis = 0 ), numpy.roll( P, -1, axis = 0 ) - P.mean( axis = 0 ) ).sum( axis = 0 )
        assert abs( area @ _np( f.normal ) - _np( f.measure ) ) < 1e-9
    c = cx.cells[ 7 ]
    assert set( c.faces( 0 ) ) == { v for f in c.children for v in f.vertices }
    assert all( c in v.cells for v in c.vertices )


if test( "centroids" ):
    # the centroids of the cells, against the first moments of the diagram
    import numpy
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 4 )
    for d in ( 2, 3 ):
        pd = PowerDiagram( rng.random( ( 30, d ) ), boundaries = _box( d ) )
        cx = pd.cell_complex()
        mass, first, _ = pd.moments
        m = numpy.asarray( mass ).reshape( -1 )
        assert numpy.allclose( _np( cx.cells.centroids ), numpy.asarray( first ).reshape( -1, d ) / m[ :, None ], atol = 1e-9 )


if test( "raw_cells" ):
    # a complex of cells that are not those of a diagram: the walls are found from the faces themselves
    import numpy
    from sdot.Cell import Cell
    from sdot.CellComplex import CellComplex
    c = Cell.make_hypercube( 3, kernel_dtype = "FP64" )
    c.cut( numpy.array( [ 1.0, 1, 1 ] ), 2.5 )                # the corner ( 1, 1, 1 ), legs of 0.5
    cx = CellComplex( c )
    assert ( cx.nb_faces( 0 ), cx.nb_faces( 1 ), cx.nb_faces( 2 ), cx.nb_faces( 3 ) ) == ( 10, 15, 7, 1 )
    assert len( cx.walls[ 1 ] ) == 7
    assert abs( _np( cx.cells.measures )[ 0 ] - ( 1 - 0.5 ** 3 / 6 ) ) < 1e-12


if test( "periodic" ):
    # the complex of the QUOTIENT: a torus, a Klein bottle, a Moebius strip ( walls on two sides ), a cone ( a disc ), a cylinder
    # in 3D -- their Euler characteristics, the measures, and on grids the exact numbers of faces ( a degenerate diagram, where the
    # folded cells carry points inside their edges )
    import numpy
    from math import comb
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    flip, quarter = numpy.diag( [ 1, -1 ] ), numpy.array( [ [ 0, -1 ], [ 1, 0 ] ] )
    groups = { "torus": ( True, 0 ), "klein": ( [ ( flip, [ 1, 1 ] ), [ 0, 1 ] ], 0 ), "moebius": ( [ ( flip, [ 1, 1 ] ) ], 0 ),
               "cone": ( [ ( quarter, [ 0, 0 ] ) ], 1 ) }
    rng = numpy.random.default_rng( 5 )
    for kd in ( "FP64", "FP32" ):
        set_kernel_dtype( kd )
        for name, ( per, chi ) in groups.items():
            pd = PowerDiagram( rng.random( ( 50, 2 ) ), 1e-3 * rng.random( 50 ), boundaries = _box( 2 ), periodicity = per )
            cx = _check( pd, euler = chi, periodic = True )
            assert abs( _np( cx.cells.measures ).sum() - 1 ) < 1e-6
            assert numpy.allclose( _np( cx.cells.measures ), numpy.asarray( _np( pd.measures ) ).reshape( -1 ), atol = 1e-6 )
            # across each facet, the seed and the image of the other: the facet is ( almost: the weights ) their bisector, and
            # its normal goes from one to the other
            sv = _np( cx.facets.seed_vectors )
            inner = cx.facets.sides[ :, 1 ] >= 0
            cos = ( sv[ inner ] * _np( cx.facets.normals )[ inner ] ).sum( axis = 1 ) / numpy.linalg.norm( sv[ inner ], axis = 1 )
            assert numpy.allclose( cos, 1, atol = 1e-4 ), ( name, cos.min() )
            assert ( sv[ ~inner ] == 0 ).all()
        for d in ( 2, 3 ):
            _check( PowerDiagram( rng.random( ( 40, d ) ), boundaries = _box( d ), periodicity = True ), euler = 0, periodic = True )
            cx = _check( PowerDiagram( rng.random( ( 40, d ) ), boundaries = _box( d ), periodicity = [ True ] + [ False ] * ( d - 1 ) ),
                         euler = 0, periodic = True )
            assert ( cx.facets.sides[ :, 1 ] < 0 ).any()          # the walls of a cylinder
            for m in ( 3, 4 ):
                g = ( numpy.stack( numpy.meshgrid( *[ numpy.arange( m ) + 0.5 ] * d, indexing = "ij" ), -1 ).reshape( -1, d ) ) / m
                cx = _check( PowerDiagram( g, boundaries = _box( d ), periodicity = True ), euler = 0, periodic = True )
                assert [ cx.nb_faces( k ) for k in range( d + 1 ) ] == [ comb( d, k ) * m ** d for k in range( d + 1 ) ], ( kd, d, m )
    set_kernel_dtype( "FP64" )


if test( "periodic_small" ):
    # two seeds on a torus: cells that meet themselves across the period, and facets between images of one seed
    import numpy
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 6 )
    for d in ( 2, 3 ):
        for _ in range( 4 ):
            pd = PowerDiagram( rng.random( ( 2, d ) ), boundaries = _box( d ), periodicity = True )
            cx = _check( pd, euler = 0, periodic = True )
            assert ( cx.facets.sides[ :, 0 ] == cx.facets.sides[ :, 1 ] ).any()
            if d == 2:
                # each facet once, on two sides of cells ( twice the same cell when it meets itself ): twice the total length of
                # the facets is the sum of the perimeters of the cells
                cells = pd.cells
                per = sum( numpy.linalg.norm( numpy.roll( cells.vertices( i ), -1, axis = 0 ) - cells.vertices( i ), axis = 1 ).sum()
                           for i in range( 2 ) )
                assert abs( 2 * _np( cx.facets.measures ).sum() - per ) < 1e-9


if test( "simplices_and_distances" ):
    # the simplices of each face add up to its measure; `locate` against the smallest power distance ( over the images on a
    # torus ); the distance to the boundary of the cell against the planes of `pd.cells`
    import math
    import numpy
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 7 )

    def volumes( pts ):
        E = pts[ :, 1: ] - pts[ :, :1 ]
        return numpy.sqrt( numpy.maximum( numpy.linalg.det( numpy.einsum( "mid,mjd->mij", E, E ) ), 0 ) ) / math.factorial( E.shape[ 1 ] )

    for per in ( None, True ):
        for d in ( 2, 3 ):
            p, w = rng.random( ( 60, d ) ), 1e-3 * rng.random( 60 )
            pd = PowerDiagram( p, w, boundaries = _box( d ), periodicity = per )
            cx = pd.cell_complex()
            for k in range( 1, d + 1 ):
                S, owner = cx.simplices( k )
                S = _np( S )
                assert S.shape[ 1: ] == ( k + 1, d )
                sums = numpy.bincount( owner, weights = volumes( S ), minlength = cx.nb_faces( k ) )
                assert numpy.allclose( sums, _np( cx.faces( k ).measures ), atol = 1e-12 ), ( per, d, k )
            assert numpy.allclose( volumes( cx.cells[ 3 ].simplices() ).sum(), _np( cx.cells[ 3 ].measure ) )

            X = rng.random( ( 2000, d ) )
            cells, moved = cx.locate( X )
            moved = _np( moved )
            shifts = numpy.zeros( ( 1, d ) ) if per is None else numpy.stack( numpy.meshgrid( *[ [ -1, 0, 1 ] ] * d, indexing = "ij" ), -1 ).reshape( -1, d )
            pw = ( ( ( X[ :, None, None, : ] - p[ None, None ] - shifts[ None, :, None ] ) ** 2 ).sum( -1 ) - w ).min( axis = 1 )
            assert ( cells == pw.argmin( axis = 1 ) ).all()
            dist, facets = cx.boundary_distances( X )
            dist = _np( dist )
            assert ( dist >= -1e-12 ).all() and ( facets >= 0 ).all()
            planes = [ pd.cells._planes_of( pd.cells._item( i ) ) for i in range( 60 ) ]
            ref = numpy.array( [ min( o - n @ moved[ i ] for n, o in zip( *planes[ cells[ i ] ] ) if numpy.abs( n ).sum() ) for i in range( 300 ) ] )
            assert numpy.allclose( dist[ :300 ], ref, atol = 1e-12 )
            # the nearest facet is a facet of that cell, at that distance from the point
            assert all( cells[ i ] in [ c.seed for c in cx.facets[ int( facets[ i ] ) ].cells ] for i in range( 300 ) )

    # outside the domain: no cell
    pd = PowerDiagram( rng.random( ( 20, 2 ) ), boundaries = _box( 2 ) )
    cells, moved = pd.cell_complex().locate( numpy.array( [ [ 1.5, 0.5 ], [ 0.5, 0.5 ] ] ) )
    assert cells[ 0 ] == -1 and cells[ 1 ] >= 0


if test( "consistent_connectivity" ):
    # grids nudged by 0, 1e-12, 1e-9, 1e-6, 1e-3 of a cell: the cells of an FP32 diagram disagree on what they share there ( a facet on
    # one side only, measured up to 5e-5 of a cell ); those of the complex are cut with canonical predicates and glued by their keys --
    # the Euler characteristic of a box ( 1 ) and of a torus ( 0 ), the copies of each facet equal, and nothing for the check to report,
    # in 2D, 3D and 4D
    import numpy
    import warnings
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    set_kernel_dtype( "FP32" )
    rng = numpy.random.default_rng( 8 )
    for d, m, jitters in ( ( 2, 30, ( 0.0, 1e-12, 1e-9, 1e-6, 1e-3 ) ), ( 3, 8, ( 0.0, 1e-12, 1e-9, 1e-6, 1e-3 ) ), ( 4, 3, ( 0.0, 1e-9 ) ) ):
        g = ( numpy.stack( numpy.meshgrid( *[ numpy.arange( m ) + 0.5 ] * d, indexing = "ij" ), -1 ).reshape( -1, d ) ) / m
        for jitter in jitters:
            p = ( g + jitter / m * rng.standard_normal( g.shape ) ) % 1.0
            for per in ( None, True ):
                with warnings.catch_warnings( record = True ) as caught:
                    warnings.simplefilter( "always" )
                    cx = PowerDiagram( p, boundaries = _box( d ), periodicity = per ).cell_complex()
                nb = [ cx.nb_faces( k ) for k in range( d ) ] + [ int( ( numpy.diff( cx.cells.incident( 0 )[ 0 ] ) > 0 ).sum() ) ]
                chi = sum( ( -1 ) ** k * nb[ k ] for k in range( d + 1 ) )
                assert chi == ( 0 if per else 1 ) and cx.is_consistent, ( d, jitter, per, nb, cx.diagnostics )
                assert cx.diagnostics[ "facet_mismatch" ] < 1e-9 and not caught, ( d, jitter, per, nb, cx.diagnostics )
    set_kernel_dtype( "FP64" )


if test( "canonical_measures" ):
    # `predicates = "canonical"`: the cells of the canonical predicates are those of the usual cut ( in FP64 ) -- on random seeds with
    # weights, on nudged grids, in a box and on a torus, 2D to 4D -- and they tile the domain
    import numpy
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 12 )
    for d, n in ( ( 2, 200 ), ( 3, 100 ), ( 4, 60 ) ):
        m = 4 if d < 4 else 3
        g = ( numpy.stack( numpy.meshgrid( *[ numpy.arange( m ) + 0.5 ] * d, indexing = "ij" ), -1 ).reshape( -1, d ) ) / m
        for p, w in ( ( rng.random( ( n, d ) ), 1e-3 * rng.random( n ) ), ( g + 1e-9 * rng.standard_normal( g.shape ), None ) ):
            for per in ( None, True ):
                kw = dict( boundaries = _box( d ), periodicity = per )
                fast = numpy.asarray( PowerDiagram( p, w, kernel_dtype = "FP64", **kw ).measures )
                pd = PowerDiagram( p, w, predicates = "canonical", **kw )
                assert pd.kernel_dtype == "FP64"
                canon = numpy.asarray( pd.measures )
                assert numpy.allclose( canon, fast, atol = 1e-12 ), ( d, per, abs( canon - fast ).max() )
                assert abs( canon.sum() - 1 ) < 1e-12


if test( "contraction" ):
    # a degenerate point blown up by the tie-breaking: with `contract = False`, the complex of the perturbed diagram -- in general
    # position, every vertex in exactly `d + 1` cells or walls, faces of zero measure included --; contracted ( the default ), the
    # exact faces of the grid. Both consistent, with the same cells
    import numpy
    from math import comb
    from sdot.PowerDiagram import PowerDiagram
    for d in ( 2, 3 ):
        m = 4
        g = ( numpy.stack( numpy.meshgrid( *[ numpy.arange( m ) + 0.5 ] * d, indexing = "ij" ), -1 ).reshape( -1, d ) ) / m
        pd = PowerDiagram( g, boundaries = _box( d ) )
        cx = _check( pd )
        assert [ cx.nb_faces( k ) for k in range( d + 1 ) ] == [ comb( d, k ) * m ** k * ( m + 1 ) ** ( d - k ) for k in range( d + 1 ) ]
        px = pd.cell_complex( contract = False )
        assert px.is_consistent
        nb = [ px.nb_faces( k ) for k in range( d + 1 ) ]
        assert nb[ 0 ] > cx.nb_faces( 0 ) and sum( ( -1 ) ** k * nb[ k ] for k in range( d + 1 ) ) == 1
        # ( each vertex in exactly `d + 1` cells or walls: its facets are `d + 1 choose 2`, its cells plus its walls `d + 1` )
        cells = numpy.diff( px.incident( 0, d )[ 0 ] )
        sides = px.facets.sides
        vo, vf = px.incident( 0, d - 1 )
        walls = numpy.array( [ len( numpy.unique( sides[ vf[ vo[ v ] : vo[ v + 1 ] ], 1 ][ sides[ vf[ vo[ v ] : vo[ v + 1 ] ], 1 ] < 0 ] ) )
                               for v in range( nb[ 0 ] ) ] )
        assert ( cells + walls == d + 1 ).all()
        assert numpy.allclose( _np( px.cells.measures ), _np( cx.cells.measures ) )


if test( "higher_dimensions" ):
    # in 4D and 5D, where a vertex of a grid is in 2^d cells and the faces of a cell are no longer those of a simple polytope: the
    # Euler characteristic and the measures on random seeds, and the exact numbers of faces of grids, in a box and on a torus
    import numpy
    from math import comb
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 11 )
    for d in ( 4, 5 ):
        n = 60 if d == 4 else 30
        _check( PowerDiagram( rng.random( ( n, d ) ), 1e-3 * rng.random( n ), boundaries = _box( d ) ) )
        _check( PowerDiagram( rng.random( ( n, d ) ), boundaries = _box( d ), periodicity = True ), euler = 0, periodic = True )
        m = 2
        g = ( numpy.stack( numpy.meshgrid( *[ numpy.arange( m ) + 0.5 ] * d, indexing = "ij" ), -1 ).reshape( -1, d ) ) / m
        cx = _check( PowerDiagram( g, boundaries = _box( d ) ) )
        assert [ cx.nb_faces( k ) for k in range( d + 1 ) ] == [ comb( d, k ) * m ** k * ( m + 1 ) ** ( d - k ) for k in range( d + 1 ) ]
        cx = _check( PowerDiagram( g, boundaries = _box( d ), periodicity = True ), euler = 0, periodic = True )
        assert [ cx.nb_faces( k ) for k in range( d + 1 ) ] == [ comb( d, k ) * m ** d for k in range( d + 1 ) ]
        assert numpy.allclose( _np( cx.faces( 1 ).measures ), 1 / m )


if test( "differentiable_geometry" ):
    # the numbers of the complex are tensors of the framework, differentiable with respect to the positions and the weights: against
    # finite differences ( jax, in doubles ), in a box and on a torus, in 2D, 3D and 4D -- and under `jax.jit`, with the topology built outside
    from errand import has_tag
    if not has_tag( "driver=jax" ):
        skip( "the jax driver only" )
    import numpy
    import jax
    import jax.numpy as jnp
    jax.config.update( "jax_enable_x64", True )
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 9 )
    for per in ( None, True ):
        for d in ( 2, 3, 4 ):
            n = 20
            p0, w0 = rng.random( ( n, d ) ), 1e-3 * rng.random( n )
            cx = PowerDiagram( jnp.asarray( p0 ), jnp.asarray( w0 ), boundaries = _box( d ), periodicity = per ).cell_complex()
            X = jnp.asarray( rng.random( ( 30, d ) ) )
            c = { k: jnp.asarray( rng.standard_normal( cx.nb_faces( k ) ) ) for k in range( d + 1 ) }
            losses = {
                "measures"    : lambda g: sum( ( g.measures( k ) * c[ k ] ).sum() for k in range( 1, d + 1 ) ),
                "centroids"   : lambda g: sum( ( g.centroids( k ) ** 2 * c[ k ][ :, None ] ).sum() for k in range( 1, d + 1 ) ),
                "normals"     : lambda g: ( g.normals ** 3 ).sum(),
                "seed_vectors": lambda g: ( g.seed_vectors ** 2 * c[ d - 1 ][ :, None ] ).sum(),
                "vertices"    : lambda g: ( g.vertex_positions ** 2 * c[ 0 ][ :, None ] ).sum(),
                "simplices"   : lambda g: ( g.simplices()[ 0 ] ** 3 ).sum(),
                "distances"   : lambda g: jnp.nansum( g.boundary_distances( X )[ 0 ] ** 2 ),
            }
            for name, f in losses.items():
                loss = lambda p, w: f( cx.geometry( p, w ) )
                gp, gw = jax.grad( loss, argnums = ( 0, 1 ) )( jnp.asarray( p0 ), jnp.asarray( w0 ) )
                dp, dw, eps = rng.standard_normal( ( n, d ) ), 1e-3 * rng.standard_normal( n ), 1e-6
                fd = ( loss( jnp.asarray( p0 + eps * dp ), jnp.asarray( w0 + eps * dw ) )
                       - loss( jnp.asarray( p0 - eps * dp ), jnp.asarray( w0 - eps * dw ) ) ) / ( 2 * eps )
                an = float( ( gp * dp ).sum() + ( gw * dw ).sum() )
                assert abs( an - float( fd ) ) <= 1e-5 * max( abs( float( fd ) ), 1.0 ), ( per, d, name, an, float( fd ) )
            # under a jit: new positions, the same topology
            f = jax.jit( jax.value_and_grad( lambda p: cx.geometry( p, jnp.asarray( w0 ) ).measures( d - 1 ).sum() ) )
            v0, g0 = f( jnp.asarray( p0 ) )
            v1, _ = f( jnp.asarray( p0 + 1e-5 ) )
            assert numpy.isfinite( float( v0 ) ) and numpy.isfinite( float( v1 ) ) and g0.shape == ( n, d )
    set_kernel_dtype( "FP64" )


if test( "torch_tensors" ):
    # with torch positions, the numbers are torch tensors that `backward` reaches
    import importlib.util
    if importlib.util.find_spec( "torch" ) is None:
        skip( "torch is not installed" )
    import numpy
    import torch
    from sdot.PowerDiagram import PowerDiagram
    p = torch.tensor( numpy.random.default_rng( 10 ).random( ( 20, 3 ) ), dtype = torch.float64, requires_grad = True )
    cx = PowerDiagram( p, boundaries = _box( 3 ) ).cell_complex()
    m = cx.facets.measures
    assert isinstance( m, torch.Tensor ) and m.requires_grad, ( type( m ), getattr( m, 'requires_grad', None ) )
    ( m.sum() + cx.cells.centroids.sum() ).backward()
    assert p.grad is not None and torch.isfinite( p.grad ).all()
