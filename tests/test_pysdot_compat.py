from errand import test

# `sdot.pysdot`, the compatibility layer for the code written for pysdot. The REFERENCE values below were computed by the
# real pysdot ( 2025-07 version, github.com/sd-ot/pysdot ) on the same inputs -- except where pysdot was found wrong
# ( periodicity, position derivatives: see those tests ), checked against Monte-Carlo or finite differences instead.

POS = [ [ 0.13, 0.21 ], [ 0.71, 0.33 ], [ 0.45, 0.62 ], [ 0.88, 0.87 ], [ 0.27, 0.79 ], [ 0.6, 0.08 ], [ 0.93, 0.5 ] ]
W = [ 0.004, 0.0, 0.008, 0.002, 0.006, 0.001, 0.003 ]


def _setup():
    import numpy
    import sdot.pysdot
    sdot.pysdot.install()
    from sdot.pysdot.domain_types import ConvexPolyhedraAssembly
    dom = ConvexPolyhedraAssembly()
    dom.add_box( [ 0, 0 ], [ 1, 1 ], 3.0 )
    return numpy, numpy.array( POS ), numpy.array( W ), dom


if test( "integrals_centroids_moments" ):
    np, pos, w, dom = _setup()
    from pysdot import PowerDiagram                       # ( through `install` )
    pd = PowerDiagram( pos, w, dom )
    assert np.allclose( pd.integrals(), [ 0.527308625293, 0.384921411707, 0.507544712409, 0.364500113409, 0.573968366281, 0.36371051716, 0.278046253741 ], atol = 1e-10 )
    assert np.allclose( pd.centroids().ravel(), [ 0.17366260166, 0.24397281016, 0.730948929284, 0.304780777051, 0.470434657673, 0.598982406531, 0.804912683183,
                                                  0.851611304399, 0.220313579658, 0.803528263064, 0.610742777384, 0.106711125775, 0.885911358466, 0.502074207348 ], atol = 1e-10 )
    assert np.allclose( pd.second_order_moments(), [ 0.017906762992, 0.011519464194, 0.016308868931, 0.009939936446, 0.02270014314, 0.012304196334, 0.006132813072 ], atol = 1e-10 )

if test( "der_integrals_wrt_weights" ):
    np, pos, w, dom = _setup()
    from sdot.pysdot import PowerDiagram
    m = PowerDiagram( pos, w, dom ).der_integrals_wrt_weights()
    assert m.error == 0
    assert list( m.m_offsets ) == [ 0, 5, 10, 16, 20, 24, 27, 31 ]
    assert list( m.m_columns ) == [ 0, 1, 2, 4, 5, 0, 1, 2, 5, 6, 0, 1, 2, 3, 4, 6, 2, 3, 4, 6, 0, 2, 3, 4, 0, 1, 5, 1, 2, 3, 6 ]
    assert np.allclose( m.m_values, [ 2.336417810059, -3.019300694194e-04, -0.906567987895, -0.4313236322556, -0.9982242598392, -3.019300694192e-04, 7.648320513774,
        -1.459053549538, -3.512412746586, -2.676552287582, -0.906567987895, -1.459053549538, 7.112656634299, -0.9311054102544, -3.517081428388, -0.2988482582233,
        -0.9311054102544, 2.240150664326, -0.1762017462278, -1.132843507844, -0.4313236322556, -3.517081428388, -0.1762017462278, 4.124606806872, -0.9982242598392,
        -3.512412746586, 4.510637006425, -2.676552287582, -0.2988482582233, -1.132843507844, 4.108244053648 ], atol = 1e-10 )
    assert np.allclose( m.v_values, PowerDiagram( pos, w, dom ).integrals() )
    # an empty cell and `stop_if_void`
    w2 = w.copy()
    w2[ 0 ] = -1
    assert PowerDiagram( pos, w2, dom ).der_integrals_wrt_weights( stop_if_void = True ).error == 1

if test( "domains" ):
    np, pos, w, dom = _setup()
    from sdot.pysdot import PowerDiagram, ScaledImage, ConvexPolyhedraAssembly
    # boxes that overlap and a `sub_box`: an image on the grid of their coordinates
    dd = ConvexPolyhedraAssembly()
    dd.add_box( [ 0, 0 ], [ 1, 1 ] )
    dd.sub_box( [ 0.5, 0.25 ], [ 1.0, 0.5 ], 0.5 )
    dd.add_box( [ 0, 0 ], [ 0.3, 1 ], 1.0 )
    assert np.isclose( dd.measure(), 1 - 0.5 * 0.5 * 0.25 + 0.3 )
    assert dd.coeff_at( [ 0.1, 0.1 ] ) == 2 and dd.coeff_at( [ 0.7, 0.3 ] ) == 0.5 and not dd.contains( [ 2, 2 ] )
    assert np.allclose( PowerDiagram( pos, w, dd ).integrals(), [ 0.324088947796, 0.089062435614, 0.182097753451, 0.121500037803, 0.326681356248, 0.120876020872, 0.073193448217 ], atol = 1e-10 )
    # an image, `img[ y, x ]`
    img = np.array( [ [ 1.0, 0.5, 2.0 ], [ 0.25, 1.5, 1.0 ] ] )
    assert np.allclose( PowerDiagram( pos * [ 2, 1 ], w, ScaledImage( [ 0, 0 ], [ 2, 1 ], img ) ).integrals(),
                        [ 0.327603682644, 0.425323993113, 0.464705752001, 0.226265491063, 0.140397024553, 0.123422909147, 0.375614480813 ], atol = 1e-10 )
    # simplices: a mesh constant on each one
    ds = ConvexPolyhedraAssembly()
    ds.add_simplex( [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ] ], 2.0 )
    ds.add_simplex( [ [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ], 1.0 )
    assert np.allclose( PowerDiagram( pos, w, ds ).integrals(), [ 0.351539083529, 0.1750935449, 0.23707765601, 0.121500037803, 0.2823544865, 0.239753106678, 0.09268208458 ], atol = 1e-10 )
    # one convex polyhedron ( points and outward normals ): the triangle x, y >= 0, x + y <= 1, which pysdot left empty
    dp = ConvexPolyhedraAssembly()
    dp.add_convex_polyhedron( [ [ 0, 0, -1, 0 ], [ 0, 0, 0, -1 ], [ 1, 0, 1, 1 ] ], 1.5 )
    ip = PowerDiagram( pos * 0.5, w, dp ).integrals()
    assert np.isclose( dp.measure(), 0.75 ) and np.isclose( ip.sum(), 0.75 )
    dt = ConvexPolyhedraAssembly()
    dt.add_simplex( [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ] ], 1.5 )
    assert np.allclose( ip, PowerDiagram( pos * 0.5, w, dt ).integrals() )

if test( "in_ball" ):
    np, pos, w, dom = _setup()
    from sdot.pysdot import PowerDiagram
    from sdot.pysdot.radial_funcs import RadialFuncInBall
    pd = PowerDiagram( pos, w + 0.02, dom, RadialFuncInBall() )
    assert np.allclose( pd.integrals(), [ 0.217650577302, 0.185886723162, 0.246385941744, 0.191960068024, 0.22660613755, 0.163677831591, 0.168391429596 ], atol = 1e-10 )
    m = pd.der_integrals_wrt_weights()
    assert list( m.m_columns ) == [ 0, 1, 5, 6, 2, 4, 3, 2, 4, 1, 5, 1, 6 ]
    assert np.allclose( m.m_values, [ 7.699524716108, 8.469237101346, -0.472002168937, -0.499845171826, 8.628948334006, -1.308325586917, 6.032928398148,
                                      -1.308325586917, 8.531951890669, -0.472002168937, 6.035661181725, -0.499845171826, 5.720281495663 ], atol = 1e-9 )

if test( "periodicity" ):
    # pysdot loses mass there ( its integrals sum to 2.53 out of 3 ): the reference is a Monte-Carlo on the torus
    np, pos, w, dom = _setup()
    from sdot.pysdot import PowerDiagram
    pd = PowerDiagram( pos, w, dom )
    shifts = [ [ 1, 0 ], [ -1, 0 ], [ 0, 1 ], [ 0, -1 ], [ 1, 1 ], [ -1, -1 ], [ 1, -1 ], [ -1, 1 ] ]
    for t in shifts:
        pd.add_replication( t )
    integrals = pd.integrals()
    x = np.random.default_rng( 0 ).uniform( 0, 1, ( 200000, 2 ) )
    d2 = np.min( [ ( ( x[ :, None, : ] - ( pos + t )[ None ] ) ** 2 ).sum( 2 ) - w for t in [ [ 0, 0 ] ] + shifts ], axis = 0 )
    mc = np.bincount( d2.argmin( 1 ), minlength = len( pos ) ) / len( x ) * 3
    assert np.isclose( integrals.sum(), 3 ) and np.allclose( integrals, mc, atol = 6e-3 )
    m = pd.der_integrals_wrt_weights()
    assert np.allclose( np.add.reduceat( m.m_values, m.m_offsets[ :-1 ].astype( int ) ), 0, atol = 1e-12 )    # ( the rows sum to 0 )

if test( "cell_polyhedra_and_vtk" ):
    np, pos, w, dom = _setup()
    import os
    import tempfile
    from sdot.pysdot import PowerDiagram
    pd = PowerDiagram( pos, w, dom )
    off, coords = pd.cell_polyhedra()
    assert len( off ) == len( pos ) + 1 and off[ -1 ] == len( coords )
    for i in range( len( pos ) ):            # counter clockwise: the shoelace formula gives the area ( density 3 )
        c = coords[ int( off[ i ] ) : int( off[ i + 1 ] ) ]
        area = 0.5 * np.sum( c[ :, 0 ] * np.roll( c[ :, 1 ], -1 ) - np.roll( c[ :, 0 ], -1 ) * c[ :, 1 ] )
        assert np.isclose( 3 * area, pd.integrals()[ i ] )
    cells, types, points = pd.vtk_mesh_data( 0.1 )
    assert len( types ) == len( pos ) and points.shape[ 1 ] == 3
    with tempfile.TemporaryDirectory() as d:
        pd.display_vtk( os.path.join( d, "sub", "pd.vtk" ), points = True, centroids = True )
        txt = open( os.path.join( d, "sub", "pd.vtk" ) ).read()
        assert f"CELL_TYPES { len( pos ) + 2 * len( pos ) }" in txt and "SCALARS num" in txt

if test( "three_d" ):
    import numpy as np
    from sdot.pysdot import PowerDiagram, OptimalTransport
    from sdot.pysdot.domain_types import ConvexPolyhedraAssembly
    pos = np.array( [ [ 0.2, 0.3, 0.2 ], [ 0.7, 0.2, 0.8 ], [ 0.4, 0.8, 1.2 ], [ 0.9, 0.7, 1.6 ], [ 0.3, 0.4, 1.8 ] ] )
    w = np.array( [ 0.01, 0.0, 0.02, 0.005, 0.0 ] )
    dom = ConvexPolyhedraAssembly()
    dom.add_box( [ 0, 0, 0 ], [ 1, 1, 2 ], 0.5 )
    pd = PowerDiagram( pos, w, dom )
    assert np.allclose( pd.integrals(), [ 0.232565878107, 0.236788978071, 0.22158965182, 0.129183608496, 0.179871883506 ], atol = 1e-10 )
    assert np.allclose( pd.centroids().ravel(), [ 0.379602295421, 0.500080371259, 0.277314420737, 0.676587418067, 0.326032923759, 0.759531910011, 0.388598053661,
                        0.733402213653, 1.083993123122, 0.816758440674, 0.595817901965, 1.61166237011, 0.332947564247, 0.372560277036, 1.708189840907 ], atol = 1e-10 )
    assert np.allclose( pd.second_order_moments(), [ 0.060391448363, 0.050701285541, 0.041460275503, 0.019277311933, 0.028712537886 ], atol = 1e-10 )
    m = pd.der_integrals_wrt_weights()
    assert list( m.m_columns ) == [ 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 ]
    assert np.allclose( m.m_values, [ 0.371508223684, -0.303770559211, -0.067737664474, -0.303770559211, 0.650935798917, -0.257226654007, -0.052834558879,
        -0.03710402682, -0.067737664474, -0.257226654007, 0.665193374204, -0.167545536788, -0.172683518936, -0.052834558879, -0.167545536788, 0.417851071426,
        -0.197470975759, -0.03710402682, -0.172683518936, -0.197470975759, 0.407258521515 ], atol = 1e-10 )
    poly, polygons, coords = pd.cell_polyhedra()
    assert list( poly ) == [ 0, 7, 15, 22, 29, 36 ] and len( polygons ) == 37 and polygons[ -1 ] == len( coords )
    ot = OptimalTransport( pos, domain = dom )
    ot.set_stopping_criterion( 1e-12, "max delta masses" )
    assert not ot.adjust_weights()
    assert np.allclose( ot.get_centroids().ravel(), [ 0.366350194925, 0.493418308622, 0.248694361461, 0.683092243274, 0.315052381023, 0.651567220512, 0.366347128598,
        0.729267068451, 0.943994121683, 0.781079494196, 0.58301817928, 1.505078224726, 0.303130939007, 0.379244062624, 1.650666071619 ], atol = 1e-8 )

if test( "optimal_transport" ):
    np, pos, w, dom = _setup()
    from sdot.pysdot import OptimalTransport
    from sdot.pysdot.radial_funcs import RadialFuncInBall, RadialFuncEntropy
    ot = OptimalTransport( pos, domain = dom )
    ot.set_stopping_criterion( 1e-12, "max delta masses" )
    assert ot.adjust_weights() is False
    assert np.allclose( ot.pd.integrals(), 3 / 7, atol = 1e-11 ) and ot.nb_diracs() == 7 and ot.dim() == 2
    assert np.allclose( ot.get_centroids().ravel(), [ 0.136482409928, 0.25873447055, 0.649602228215, 0.32715127674, 0.380529630109, 0.626593678986, 0.752042116753,
                                                      0.859402531224, 0.179489004776, 0.821423335358, 0.547942749017, 0.113811172596, 0.853911861203, 0.492883534546 ], atol = 1e-8 )
    assert len( ot.delta_m ) and ot.delta_m[ -1 ] < 1e-12
    # again from new positions ( Lloyd ), and given masses
    ot.set_positions( ot.get_centroids() )
    ot.set_masses( np.linspace( 1, 2, 7 ) * 3 / np.linspace( 1, 2, 7 ).sum() )
    assert not ot.adjust_weights()
    assert np.allclose( ot.pd.integrals(), ot.get_masses(), atol = 1e-11 )
    # partial: the cells in their balls `|x - p_i|^2 <= w_i`
    ob = OptimalTransport( pos, domain = dom, radial_func = RadialFuncInBall(), masses = np.full( 7, 0.05 ) )
    ob.set_stopping_criterion( 1e-12, "max delta masses" )
    assert not ob.adjust_weights()
    assert np.allclose( ob.pd.integrals(), 0.05, atol = 1e-11 )
    # entropic: the weights of pysdot, those that give its absolute masses
    oe = OptimalTransport( pos, domain = dom, radial_func = RadialFuncEntropy( 0.01 ) )
    oe.set_stopping_criterion( 1e-12, "max delta masses" )
    assert not oe.adjust_weights()
    assert np.allclose( oe.pd.integrals(), 3 / 7, atol = 1e-9 )

if test( "transport_map" ):
    np, pos, w, dom = _setup()
    from sdot.pysdot import find_transport_map
    tm = find_transport_map( pos, domain = dom )
    assert np.allclose( tm.pd.integrals(), 3 / 7, atol = 1e-6 )

if test( "der_centroids_and_integrals" ):
    # by finite differences: those of pysdot do not match them with respect to the positions
    np, pos, w, dom = _setup()
    from sdot.pysdot import PowerDiagram
    try:
        import jax                                       # ( through the automatic differentiation of sdot )
    except ImportError:
        import torch
    m = PowerDiagram( pos, w, dom ).der_centroids_and_integrals_wrt_weight_and_positions()
    from scipy.sparse import csr_matrix
    n = len( pos )
    M = csr_matrix( ( m.m_values, m.m_columns.astype( int ), m.m_offsets.astype( int ) ), shape = ( 3 * n, 3 * n ) ).toarray()

    def f( p, ww ):
        pd = PowerDiagram( p, ww, dom )
        return np.concatenate( [ pd.centroids(), pd.integrals()[ :, None ] ], 1 ).ravel()
    assert np.allclose( m.v_values, f( pos, w ) )
    h = 1e-6
    for j in range( n ):
        for c in range( 3 ):
            dp, dw = np.zeros_like( pos ), np.zeros_like( w )
            if c < 2:
                dp[ j, c ] = h
            else:
                dw[ j ] = h
            fd = ( f( pos + dp, w + dw ) - f( pos - dp, w - dw ) ) / ( 2 * h )
            assert np.allclose( M[ :, 3 * j + c ], fd, atol = 1e-7 )

if test( "install" ):
    import sys
    import sdot.pysdot
    sdot.pysdot.install()
    import pysdot
    import pysdot.domain_types
    from pysdot.radial_funcs import RadialFuncUnit
    from pysdot.PowerDiagram import PowerDiagram
    assert pysdot is sdot.pysdot and pysdot.domain_types.ConvexPolyhedraAssembly is sdot.pysdot.ConvexPolyhedraAssembly
    assert PowerDiagram is sdot.pysdot.PowerDiagram and sys.modules[ "pysdot.OptimalTransport" ].BadInitialGuess is sdot.pysdot.BadInitialGuess

if test( "polynomial_pixels" ):
    # `img[ c, y, x ]`, 6 coefficients: c0 + c1 x + c2 y + c3 x^2 + c4 x y + c5 y^2 per pixel
    np, pos, w, dom = _setup()
    from sdot.pysdot import PowerDiagram, ScaledImage
    img = np.zeros( [ 6, 2, 3 ] )
    img[ 0 ] = [ [ 1.0, 0.5, 2.0 ], [ 0.25, 1.5, 1.0 ] ]
    img[ 1 ] = 0.3
    img[ 2, 1, 2 ] = -0.2
    img[ 3 ] = 0.1
    img[ 4, 0, 1 ] = 0.4
    img[ 5 ] = 0.05
    si = ScaledImage( [ 0, 0 ], [ 1, 1 ], img )
    assert np.isclose( si.measure(), 1.225 ) and np.isclose( si.coeff_at( [ 0.6, 0.7 ] ), 1.7405 )
    assert np.allclose( PowerDiagram( pos, w, si ).integrals(), [ 0.176263258456, 0.229395133032, 0.226249607042, 0.153839915268, 0.124123799498,
                                                                  0.154076002885, 0.161052283819 ], atol = 1e-10 )

if test( "replications_that_are_not_a_periodicity" ):
    # the domain overlaps its images: the translated seeds are more seeds, as pysdot did
    np, pos, w, dom = _setup()
    from sdot.pysdot import PowerDiagram, ConvexPolyhedraAssembly
    big = ConvexPolyhedraAssembly()
    big.add_box( [ -1, -1 ], [ 2, 2 ] )
    pd = PowerDiagram( pos, w, big )
    pd.add_replication( [ 1, 0 ] )
    pd.add_replication( [ -1, 0 ] )
    assert np.allclose( pd.integrals(), [ 0.363066849052, 0.111252961608, 0.168326033817, 0.860496439157, 0.430762259943, 0.934474609118, 0.131620847304 ], atol = 1e-10 )
    assert np.allclose( pd.centroids().ravel(), [ 0.127646692241, -0.032192898712, 0.698648263096, 0.327778973368, 0.471872421278, 0.599458775466, 0.848745232132,
                                                  1.424874889443, 0.319022910239, 1.203144910636, 0.60210520856, -0.469467570997, 0.947062322667, 0.521601338988 ], atol = 1e-10 )
    m = pd.der_integrals_wrt_weights()
    assert m.m_offsets[ -1 ] == len( m.m_values ) and np.all( m.m_columns < len( pos ) )

if test( "distances_from_boundaries" ):
    np, pos, w, dom = _setup()
    from sdot.pysdot import PowerDiagram, ConvexPolyhedraAssembly
    unit = ConvexPolyhedraAssembly()
    unit.add_box( [ 0, 0 ], [ 1, 1 ] )
    pts = np.array( [ [ 0.1, 0.1 ], [ 0.5, 0.5 ], [ 0.9, 0.2 ], [ 0.3, 0.95 ], [ 0.7, 0.7 ] ] )
    pd = PowerDiagram( pos, w, unit )
    assert np.allclose( pd.distances_from_boundaries( pts ), [ -0.24648956735, -0.082287760596, -0.06276328761, -0.20780459277, -0.001608385464 ], atol = 1e-10 )
    assert np.allclose( pd.distances_from_boundaries( pts, True ), [ -0.1, -0.082287760596, -0.06276328761, -0.05, -0.001608385464 ], atol = 1e-10 )

if test( "older_call_forms_and_outputs" ):
    # `OptimalTransport( domain, radial_func )`: what the euler and crowd examples of pysdot still write
    np, pos, w, dom = _setup()
    import os
    import tempfile
    from sdot.pysdot import OptimalTransport, PowerDiagram
    from sdot.pysdot.radial_funcs import RadialFuncInBall
    ot = OptimalTransport( dom, RadialFuncInBall() )
    ot.set_positions( pos )
    ot.set_weights( np.full( 7, 0.01 ) )
    ot.set_masses( np.full( 7, 0.05 ) )
    assert not ot.adjust_weights()
    assert np.allclose( ot.pd.integrals(), 0.05, atol = 1e-10 )
    with tempfile.TemporaryDirectory() as d:
        ot.display_asy( os.path.join( d, "pd.asy" ), values = np.linspace( 0, 1, 7 ), linewidth = 0.005 )
        txt = open( os.path.join( d, "pd.asy" ) ).read()
        assert txt.count( "fill(" ) == 7 and txt.count( "draw(" ) == 7
        ot.display_vtk( os.path.join( d, "pd.vtk" ) )
    # the cells shown in their balls: the polygon of each one has the area of its integral ( density 3 )
    polys, nums = ot.pd._polygons()
    areas = [ 0.5 * abs( np.sum( p[ :, 0 ] * np.roll( p[ :, 1 ], -1 ) - np.roll( p[ :, 0 ], -1 ) * p[ :, 1 ] ) ) for p in polys ]
    assert np.allclose( 3 * np.array( areas ), 0.05, rtol = 3e-3 )
    # a domain made of pieces: each cell shown cut by each piece, as pysdot did
    from sdot.pysdot import ConvexPolyhedraAssembly
    two = ConvexPolyhedraAssembly()
    two.add_box( [ 0, 0 ], [ 0.5, 1 ] )
    two.add_box( [ 0.5, 0 ], [ 1, 0.5 ] )
    polys, nums = PowerDiagram( pos, w, two )._polygons()
    assert all( np.all( p[ :, 0 ] <= 0.5 + 1e-12 ) or np.all( p[ :, 1 ] <= 0.5 + 1e-12 ) for p in polys )
    # memoized: the same diagram is not recomputed
    pd = PowerDiagram( pos, w, dom )
    c1 = pd.centroids( 0.5 )
    assert pd._memo( "measures", None ) is pd._memo( "measures", None ) and not np.allclose( c1, pd.centroids( 0.5 ) )

if test( "pieces_with_a_hole" ):
    # two boxes and a 16-gon of coefficient -1 ( the obstacle of pysdot's crowd example ): the arrangement of the pieces in 2D
    import numpy as np
    from sdot.pysdot import PowerDiagram, ConvexPolyhedraAssembly
    pon = [ [ 1.5 + 0.25 * np.cos( a ), 0.5 + 0.25 * np.sin( a ), np.cos( a ), np.sin( a ) ] for a in 2 * np.pi * np.arange( 16 ) / 16 ]
    dom = ConvexPolyhedraAssembly()
    dom.add_box( [ 0, 0 ], [ 2, 1 ] )
    dom.add_box( [ 2, 0.4 ], [ 3, 0.6 ] )
    dom.add_convex_polyhedron( np.array( pon ), -1.0 )
    assert np.isclose( dom.measure(), 2.001087632620342 ) and not dom.contains( [ 1.5, 0.5 ] ) and dom.contains( [ 2.5, 0.5 ] )
    pos = np.array( [ [ 0.3, 0.2 ], [ 1.2, 0.5 ], [ 1.5, 0.9 ], [ 1.9, 0.3 ], [ 2.5, 0.5 ], [ 0.8, 0.7 ], [ 1.5, 0.1 ] ] )
    pd = PowerDiagram( pos, np.zeros( 7 ), dom )
    assert np.allclose( pd.integrals(), [ 0.486666666667, 0.251723928403, 0.234829930135, 0.226805456604, 0.166666666667, 0.472694805195, 0.161700178951 ], atol = 1e-10 )
    assert np.allclose( pd.centroids().ravel(), [ 0.318112633181, 0.340639269406, 1.078911381012, 0.393945280731, 1.585082851082, 0.862172096464, 1.899311537424,
                                                  0.378386829277, 2.583111111111, 0.501333333333, 0.659327519105, 0.72991464199, 1.42269784118, 0.115857379931 ], atol = 1e-10 )
    # the cells shown without the hole
    polys, nums = pd._polygons()
    assert not any( np.linalg.norm( p.mean( axis = 0 ) - [ 1.5, 0.5 ] ) < 0.2 for p in polys )
    # overlapping simplices: the coefficients add up
    dt = ConvexPolyhedraAssembly()
    dt.add_simplex( [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ] ], 2.0 )
    dt.add_simplex( [ [ 0.2, 0.2 ], [ 1, 0.2 ], [ 0.2, 1 ] ], 1.0 )
    assert np.allclose( PowerDiagram( pos * 0.5, np.zeros( 7 ), dt ).integrals(), [ 0.248333333333, 0.200832219251, 0.054001002674, 0.035286096257, 0.0,
                                                                                     0.682494318182, 0.099053030303 ], atol = 1e-10 )
