import numpy
import sdot

if __name__ == "__main__":
    numpy.random.seed( 0 )
    pts = numpy.random.random( [ 1000, 2 ] )

    f = sdot.SumOfWeightedDiracs( pts )
    o = 3
    g = sdot.PiecewiseConstantGrid( [ [ o, 1, o ], [ 1, o, 1 ], [ o, 1, o ] ] )

    ot_solver = sdot.OtSolver( sdot.Bsp( f.positions, f.weights ), g )
    ot_solver.solve( verbose = 1 )
    ot_solver.plot()

