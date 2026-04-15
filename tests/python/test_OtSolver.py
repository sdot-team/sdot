import numpy
import sdot

if __name__ == "__main__":

    numpy.random.seed( 0 )
    pts = numpy.random.random( [ 1000, 2 ] )

    # s = sdot.SplineGrid( [ 1, 2, 1 ], knots = [ [ 0, 3, 4 ] ], continuity = 1 )
    f = sdot.SumOfWeightedDiracs( pts )
    g = sdot.SplineGrid( [ [ 0, 1, 1 ], [ 0, 1, 1 ] ], continuity = 0 )

    s = sdot.OtSolver( sdot.Bsp( pts ), g )
    s.solve( verbose = 1 )
    s.plot()
