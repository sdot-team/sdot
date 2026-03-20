import sdot

f = sdot.PiecewiseAffineFunction1d( [ 0, 1 ], [ 1, 1 ] )
g = sdot.SumOfWeightedDiracs1d( [ 1, 0 ] )
o = sdot.plan( f, g )

print( o.barycenters )
print( o.distance )
print( o.cuts )
