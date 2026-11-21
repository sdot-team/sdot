"""BAKED ambient occlusion of the faces of a scene ( sdot ADR 0001 §4 ): how open the space is in front of each CORNER of
each polygon, computed once, then used by the viewer's lighting in real time.

    ao = ambient_occlusion( pos, poly_v, poly_off )      # one value per corner ( per entry of `poly_v` ), 1 = open

For each corner, a point just inside its polygon, on each SIDE of it: `rays` directions of the hemisphere ( cosine
weighted ), cast up to `radius`; a ray that meets a face at distance `t` occludes by `1 - t / radius` ( a near face more
than a far one, nothing beyond `radius`: the occlusion stays LOCAL, the inside of a closed foam is not all black ). The
value of a side is one minus the mean occlusion of its rays.

Which side: the polygons are split into PIECES ( connected through their vertices ). A closed piece ( every edge shared by
two of its polygons: a cell ) is seen from outside: the side pointing away from its centre. An open piece ( a sheet ) is
seen from both sides: the corner keeps the more open one.

The rays are cast by a C++ kernel ( `include/sdot/AmbientOcclusion.h`: a BVH over the fans of the polygons, the closest
hit ); this module prepares the samples. A numpy version ( buckets on a grid, batched tests ) took 79 s for 1e4 corners.
"""
from __future__ import annotations

import numpy as np




def _polygons( pos, poly_v, poly_off ):
    """per polygon: its centroid and its unit normal ( Newell ); per corner: its polygon"""
    nb = len( poly_off ) - 1
    lens = np.diff( poly_off )
    owner = np.repeat( np.arange( nb ), lens )
    p = pos[ poly_v ]
    nxt = np.arange( len( poly_v ) ) + 1
    nxt[ poly_off[ 1: ] - 1 ] = poly_off[ :-1 ]                     # ( the corner after the last is the first )
    q = pos[ poly_v[ nxt ] ]
    newell = np.stack( [ ( p[ :, 1 ] - q[ :, 1 ] ) * ( p[ :, 2 ] + q[ :, 2 ] ),
                         ( p[ :, 2 ] - q[ :, 2 ] ) * ( p[ :, 0 ] + q[ :, 0 ] ),
                         ( p[ :, 0 ] - q[ :, 0 ] ) * ( p[ :, 1 ] + q[ :, 1 ] ) ], axis = 1 )
    nrm = np.zeros( ( nb, 3 ) )
    np.add.at( nrm, owner, newell )
    ln = np.linalg.norm( nrm, axis = 1, keepdims = True )
    nrm = np.where( ln > 0, nrm / np.where( ln > 0, ln, 1 ), [ [ 0, 0, 1.0 ] ] )
    cen = np.zeros( ( nb, 3 ) )
    np.add.at( cen, owner, p )
    cen /= np.maximum( lens, 1 )[ :, None ]
    edge = np.linalg.norm( q - p, axis = 1 )
    return cen, nrm, owner, edge


def _triangles( poly_v, poly_off ):
    """the fans of the polygons, `( T, 3 )` indices"""
    tris = [ ]
    for k in range( 1, int( np.diff( poly_off ).max( initial = 0 ) ) - 1 ):
        p = np.nonzero( np.diff( poly_off ) > k + 1 )[ 0 ]
        o = poly_off[ p ]
        tris.append( np.stack( [ poly_v[ o ], poly_v[ o + k ], poly_v[ o + k + 1 ] ], axis = 1 ) )
    return np.concatenate( tris ) if tris else np.zeros( ( 0, 3 ), np.int64 )




def ambient_occlusion( pos, poly_v, poly_off, rays: int = 32, radius: float | None = None, seed: int = 0 ) -> np.ndarray:
    """`[ len( poly_v ) ]` float32 in [ 0, 1 ]: how open the space is in front of each corner ( see the module ).

    `pos [ n, 3 ]`, `poly_v` ( indices into `pos` ), `poly_off [ nb + 1 ]` ( the polygons' slices of `poly_v` ). `radius`:
    how far a face still occludes ( default: 4 median edge lengths )."""
    pos = np.ascontiguousarray( pos, np.float64 ).reshape( -1, 3 )
    poly_v = np.asarray( poly_v, np.int64 ).reshape( -1 )
    poly_off = np.asarray( poly_off, np.int64 ).reshape( -1 )
    if len( poly_v ) == 0:
        return np.ones( 0, np.float32 )
    cen, nrm, owner, edge = _polygons( pos, poly_v, poly_off )
    if radius is None:
        radius = 4 * float( np.median( edge[ edge > 0 ] ) ) if np.any( edge > 0 ) else 1.0
    tri = _triangles( poly_v, poly_off )
    if len( tri ) == 0:
        return np.ones( len( poly_v ), np.float32 )
    side = _sides( pos, poly_v, poly_off, cen, nrm, owner )[ owner ]
    pts = pos[ poly_v ] + 0.05 * ( cen[ owner ] - pos[ poly_v ] )     # ( each corner pulled a little into its polygon )
    spin = np.random.default_rng( seed ).uniform( 0, 2 * np.pi, len( poly_v ) )   # ( a turn per corner: noise, not bands )
    return _cast( pos, tri, pts, nrm[ owner ], spin, side, float( radius ), 1e-4 * float( radius ), int( rays ) )


def _sides( pos, poly_v, poly_off, cen, nrm, owner ) -> np.ndarray:
    """per polygon: +1 / -1, the side along its normal seen from outside of its closed piece; 0 in an open piece"""
    nb = len( poly_off ) - 1
    # the pieces: a label per vertex, the smallest of its polygons' vertices, spread until it no longer changes
    nxt = np.arange( len( poly_v ) ) + 1
    nxt[ poly_off[ 1: ] - 1 ] = poly_off[ :-1 ]
    a, b = poly_v, poly_v[ nxt ]
    lab = np.arange( len( pos ) )
    while True:
        m = np.minimum( lab[ a ], lab[ b ] )
        new = lab.copy()
        np.minimum.at( new, a, m )
        np.minimum.at( new, b, m )
        new = new[ new ]
        if np.array_equal( new, lab ):
            break
        lab = new
    piece = lab[ poly_v[ poly_off[ :-1 ] ] ]                          # ( per polygon )
    # closed: every edge of the piece met exactly twice
    e = np.sort( np.stack( [ a, b ], axis = 1 ), axis = 1 )
    _, inv, cnt = np.unique( e, axis = 0, return_inverse = True, return_counts = True )
    bad = np.zeros( len( pos ), bool )
    bad[ lab[ a[ cnt[ inv.reshape( -1 ) ] != 2 ] ] ] = True
    closed = ~bad[ piece ]
    # the centre of each piece: the mean of its polygons' centres
    tot = np.zeros( ( len( pos ), 3 ) )
    num = np.zeros( len( pos ) )
    np.add.at( tot, piece, cen )
    np.add.at( num, piece, 1 )
    mid = tot[ piece ] / num[ piece ][ :, None ]
    out = np.sign( np.einsum( "ij,ij->i", nrm, cen - mid ) )
    return np.where( closed & ( out != 0 ), out, 0.0 )


def _cast( pos, tri, pts, nrm, spin, side, radius, eps, rays ):
    import loom
    from loom.compilation.FfiCode import FfiCode
    from loom.tensor import Axis, IntTensor, RealTensor, ShapeVar
    dim = Axis( ShapeVar( 3 ), name = "dim" )
    num_node = Axis( ShapeVar( len( pos ) ), name = "num_node" )
    num_tri = Axis( ShapeVar( len( tri ) ), name = "num_tri" )
    num_corner = Axis( ShapeVar( 3 ), name = "num_corner" )
    num_sample = Axis( ShapeVar( len( pts ) ), name = "num_sample" )
    out = RealTensor[ num_sample ]()
    loom.ffi_call(
        "ambient_occlusion",
        FfiCode.inline( "sdot::ambient_occlusion::run( queue, args.inputs.pos, args.inputs.tris, args.inputs.pts, args.inputs.nrm, "
                        "args.inputs.spin, args.inputs.side, args.inputs.params, args.outputs.out );",
                        includes = [ "sdot/AmbientOcclusion.h" ] ),
        pos = RealTensor[ num_node, dim ]( pos ),
        tris = IntTensor[ num_tri, num_corner, dict( size = 32 ) ]( np.ascontiguousarray( tri, np.int32 ) ),
        pts = RealTensor[ num_sample, dim ]( np.ascontiguousarray( pts, np.float64 ) ),
        nrm = RealTensor[ num_sample, dim ]( np.ascontiguousarray( nrm, np.float64 ) ),
        spin = RealTensor[ num_sample ]( np.ascontiguousarray( spin, np.float64 ) ),
        side = RealTensor[ num_sample ]( np.ascontiguousarray( side, np.float64 ) ),
        params = RealTensor[ Axis( ShapeVar( 3 ), name = "num_param" ) ]( np.array( [ radius, eps, rays ], np.float64 ) ),
        out = loom.out( out ),
        has_dynamic_capacity = False,
    )
    return np.asarray( out, np.float32 ).reshape( -1 )


def as_codes( ao ) -> np.ndarray:
    """the uint8 form stored in pages and chunks ( 255 = open )"""
    return np.clip( np.round( np.asarray( ao, np.float64 ) * 255 ), 0, 255 ).astype( np.uint8 )
