import numpy
import sdot


if __name__ == "__main__":
    numpy.random.seed( 0 )
    pts = numpy.random.random( [ 10, 2 ] )

    # f = sdot.SumOfWeightedDiracs( [ [ 0.5, 0.5 ], [ 0.75, 0.5 ], [ 0.5, 0.75 ], ] )
    f = sdot.SumOfWeightedDiracs( pts )

    ot_solver = sdot.OtSolver(
        sdot.Bsp( f.positions, f.weights ),
        sdot.PiecewiseConstantGrid( [ [ 1 ] ] ),
    )

    pot = numpy.zeros( [ f.nb_diracs ] )

    for i in range( 10 ):
        s = ot_solver.search_dir( pot )
        ic( numpy.linalg.norm( s ) )

        pot -= s

    # 0.002966411023470022
    ot_solver.plot( pot )

