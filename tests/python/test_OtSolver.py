import numpy
import sdot

from icecream.builtins import install
install()

if __name__ == "__main__":
    f = sdot.SumOfWeightedDiracs( [ [ 0.5, 0.5 ], [ 0.75, 0.5 ], [ 0.5, 0.75 ], ] )

    ot_solver = sdot.OtSolver(
        sdot.Bsp( f.positions, f.weights ),
        sdot.PiecewiseConstantGrid( [ [ 1 ] ] ),
    )

    ot_solver.write_vtk( "yo.vtk" )
