import sdot

f = sdot.Piecewise1dAffineFunction( [ 0, 1 ], [ 1, 1 ] )
g = sdot.SumOf1dWeightedDiracs( [ 1, 0 ] )
o = sdot.plan( f, g )

print( o.barycenters )
print( o.distance )
print( o.cuts )
