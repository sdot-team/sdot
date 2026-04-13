import numpy
import sdot

if __name__ == "__main__":
    numpy.random.seed( 0 )
    pts = numpy.random.random( [ 1000, 2 ] )

    # o = 3
    # g = sdot.PiecewiseConstantGrid( [ [ o, 1, o ], [ 1, o, 1 ], [ o, 1, o ] ] )

    # f = sdot.SumOfWeightedDiracs( pts )
    # g = sdot.SplineGrid( [ [ 1, 1, 1 ], [ 1, 10, 1 ], [ 1, 1, 1 ], ], order = 1 ).normalized_version()

    # ot_solver = sdot.OtSolver( sdot.Bsp( f.positions, f.weights ), g )
    # ot_solver.solve( verbose = 1 )
    # ot_solver.plot()
    s = sdot.SplineGrid( [ [ 1, 2, 2 ], [ 10, 12, 12 ], [ 1, 2, 2 ], [ 10, 11, 11 ] ], knots = [
        [ 0, 3, 4, 5 ],
        [ 0, 3, 5 ],
    ], continuity = 1 )
    g = s.normalized_version()

    xs = numpy.linspace( 0, 5, 100 )
    from matplotlib import pyplot
    for y in numpy.linspace( 0, 5, 3 ):
        ys = [ g[ x, y ] for x in xs ]
        pyplot.plot( xs, ys )
    pyplot.show()
