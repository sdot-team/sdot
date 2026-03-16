import sdot

f = sdot.Piecewise1dAffineFunction( [ 0, 1 ], [ 1, 1 ] )
g = sdot.SumOf1dWeightedDiracs( [ 0, 1 ] )
print( sdot.barycenters( f, g ) )
print( sdot.distance( f, g ) )
