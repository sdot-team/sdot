import numpy as np
import sdot

rng = np.random.default_rng( 0 )
f = sdot.SumOfDiracs( rng.random( ( 100, 2 ) ) )       # the source: diracs ( `weights`: their masses )

targets = [
    sdot.Box(),                                          # the unit square ( the dimension comes from the source )
    sdot.Box( origin = [ -1, -1 ], frame = [ 2, 2 ] ),   # [ -1, 1 ]^2 ( `frame`: the edges, or their lengths )
    sdot.Image( rng.random( ( 64, 64 ) ) ),            # 64 x 64 pixels on [ 0, 1 ]^2
    sdot.Image( rng.random( ( 32, 64 ) ) ),            # [ 0, 0.5 ] x [ 0, 1 ]: the longest side is 1
    sdot.Image( rng.random( ( 32, 64 ) ), frame = [ 1, 1 ] ),           # [ 0, 1 ]^2: `frame`, the edges of the image
    sdot.Image( rng.random( ( 32, 64 ) ), pixel_frame = [ [ 0.1, 0 ], [ 0, 0.1 ] ] ),   # pixels of 0.1
    sdot.Polygon( [ [ 0, 0 ], [ 1, 0 ], [ 0.5, 1 ] ] ),  # a uniform density on a convex polygon
    sdot.SumOfGaussians( [ [ 0.3, 0.3 ], [ 0.7, 0.6 ] ], sigmas = [ 0.1, 0.2 ], weights = [ 1, 2 ] ),
    sdot.Mesh( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ], [ [ 0, 1, 2 ], [ 0, 2, 3 ] ],
               values = [ 0, 1, 2, 1 ] ),                # piecewise linear: one value per node
]
for g in targets:
    print( type( g ).__name__, sdot.w2( f, g ) )
