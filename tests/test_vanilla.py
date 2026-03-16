import sdot

f = sdot.Piecewise1dAffineFunction( [ 0, 1 ], [ 1, 1 ] )
g = sdot.SumOfWeightedDiracs( [ [ 0 ], [ 0 ] ] )
print( sdot.distance( f, g ) )
