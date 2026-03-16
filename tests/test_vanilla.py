import sdot

f = sdot.PiecewiseAffineFunction( [ 0, 1 ], [ 1, 1 ] )
g = sdot.SumOfWeightedDiracs( [ [ 0 ], [ 0 ] ] )
print( sdot.distances( f, g ) )
