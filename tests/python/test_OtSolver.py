import numpy
import sdot
from sdot.distributions.helpers.distribution_methods import to_tensor_list

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
    # s = sdot.SplineGrid( [ 1, 2, 1 ], knots = [ [ 0, 3, 4 ] ], continuity = 1 )
    s = sdot.SplineGrid( [ [ 1, 2 ], [ 2, 2 ], [ 0, 1 ], [ 1, 2 ] ], continuity = 0 )
    # s = sdot.SplineGrid( [ 1, 2, 1, 3 ], continuity = 0 )
    g = s.normalized_version()

    from matplotlib import pyplot
    xs = numpy.linspace( 0, 3, 100 )
    for y in numpy.linspace( 0, 1, 5 ):
        ys = [ g[ x, y ] for x in xs ]
        pyplot.plot( xs, ys )
    pyplot.show()
