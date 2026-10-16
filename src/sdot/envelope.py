"""The envelope of a codimension >= 1 mesh: the support of its smoothed density ( `src/sdot/include/sdot/Envelope.h` ).

`M` is given by `nodes [ N, D ]` and `simplices [ E, K ]`: segments ( `K = 2` ) in 2D or 3D, or triangles ( `K = 3` ) in 3D, with
`rho [ N ]` at the nodes. The envelope is the generalized Voronoi diagram of the elements of `M` ( its simplices and all their
faces ) in the box `[ lo, hi ]`, as a soup of simplices where the distance `d` to `M` and the density `rho` ( at the closest point )
are affine. The smoothed density at `s` is `w rho exp( -d^2 / s^2 )`: `w = 1` but on the fans around the segments of lines in 3D,
where it makes the limit `s -> 0` exact.
"""

import numpy

from loom.tensor import Axis, IntTensor, RealTensor, ShapeVar


def envelope( nodes, simplices, rho = None, lo = None, hi = None, alpha = 15.0, margin = 0.25, raw = False ):
    """see the module docstring. `lo`, `hi`: the box ( default: the box of `M` widened by `margin` times its diagonal ), `alpha`:
    the widest sector of the fans, in degrees. A dict: `pts [ n, D + 1, D ]`, `d [ n, D + 1 ]`, `rho [ n, D + 1 ]`, `w [ n ]`,
    `nb_candidates`, `nb_pieces`. `raw`: the candidates themselves, overlapping ( a check )"""
    import loom
    from loom.compilation.FfiCode import FfiCode
    nd = numpy.ascontiguousarray( nodes, dtype = float )
    sx = numpy.ascontiguousarray( simplices, dtype = numpy.int32 )
    D, K = nd.shape[ 1 ], sx.shape[ 1 ]
    if ( D, K ) not in ( ( 2, 2 ), ( 3, 2 ), ( 3, 3 ) ):
        raise ValueError( f"envelope: segments in 2D or 3D, or triangles in 3D ( got { K } nodes per simplex in { D }D )" )
    rh = numpy.ones( len( nd ) ) if rho is None else numpy.ascontiguousarray( rho, dtype = float )
    if lo is None or hi is None:
        a, b = nd.min( 0 ), nd.max( 0 )
        m = margin * numpy.linalg.norm( b - a )
        lo = a - m if lo is None else lo
        hi = b + m if hi is None else hi
    params = numpy.concatenate( [ numpy.asarray( lo, dtype = float ), numpy.asarray( hi, dtype = float ), [ numpy.radians( alpha ), float( raw ) ] ] )

    num_node = Axis( ShapeVar( len( nd ) ), name = "num_node" )
    num_elem = Axis( ShapeVar( len( sx ) ), name = "num_elem" )
    dim = Axis( ShapeVar( D ), name = "dim" )
    cap = 64 * len( sx ) * ( 4 if D == 3 else 1 )
    while True:
        num_out = Axis( ShapeVar( cap ), name = "num_out" )
        num_corner = Axis( ShapeVar( D + 1 ), name = "num_corner" )
        pts = RealTensor[ num_out, num_corner, dim ]()
        d = RealTensor[ num_out, num_corner ]()
        r = RealTensor[ num_out, num_corner ]()
        w = RealTensor[ num_out ]()
        stats = IntTensor[ Axis( ShapeVar( 3 ), name = "num_stat" ) ]()
        loom.ffi_call(
            "codim_envelope",
            FfiCode.inline( f"sdot::envelope::run<{ D },{ K }>( queue, args.inputs.nodes, args.inputs.simplices, args.inputs.rho, "
                            "args.inputs.params, args.outputs.points, args.outputs.dists, args.outputs.rhos, args.outputs.weights, args.outputs.stats );",
                            includes = [ "sdot/Envelope.h" ] ),
            nodes = RealTensor[ num_node, dim ]( nd ),
            simplices = IntTensor[ num_elem, Axis( ShapeVar( K ), name = "num_vert" ), dict( size = 32 ) ]( sx ),
            rho = RealTensor[ num_node ]( rh ),
            params = RealTensor[ Axis( ShapeVar( len( params ) ), name = "num_param" ) ]( params ),
            points = loom.out( pts ), dists = loom.out( d ), rhos = loom.out( r ), weights = loom.out( w ), stats = loom.out( stats ),
            has_dynamic_capacity = False,
        )
        st = numpy.asarray( stats ).reshape( -1 ).astype( int )
        if st[ 0 ] <= cap:
            break
        cap = int( st[ 0 ] )                                    # ( the envelope did not fit: redone with room for it )
    n = st[ 0 ]
    return dict( pts = numpy.asarray( pts ).reshape( cap, D + 1, D )[ :n ], d = numpy.asarray( d ).reshape( cap, D + 1 )[ :n ],
                 rho = numpy.asarray( r ).reshape( cap, D + 1 )[ :n ], w = numpy.asarray( w ).reshape( cap )[ :n ],
                 nb_candidates = int( st[ 1 ] ), nb_pieces = int( st[ 2 ] ) )
