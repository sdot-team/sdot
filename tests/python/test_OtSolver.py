import numpy
import sdot

if __name__ == "__main__":
    numpy.random.seed( 0 )
    pts = numpy.random.random( [ 10000, 2 ] )

    # f = sdot.SumOfWeightedDiracs( [ [ 0.5, 0.5 ], [ 0.75, 0.5 ], [ 0.5, 0.75 ], ] )
    f = sdot.SumOfWeightedDiracs( pts )

    ot_solver = sdot.OtSolver(
        sdot.Bsp( f.positions, f.weights ),
        sdot.PiecewiseConstantGrid( [ [ 1 ] ] ),
    )

    ot_solver.solve()

    # pot = numpy.zeros( [ f.nb_diracs ] )
    # gc_eigen( ot_solver, pot, nb_to_keep = 4 )

    # 0.002966411023470022 sans scaling
    # avec t directions 0.15815169659806913 0.05461503859992672 0.03473914792301476 0.009736285477986202 0.005886671527922016 0.0016232694072783893 0.00014316491531729186 5.422471915076703e-05 2.593768513429894e-06  4.411554590049321e-12
    # avec 2 directions 0.15815169659806913 0.05461503859992672 0.03473914792301476 0.008562565011766159 0.005599299799189152 0.002140419003793201  0.0010762166398478594 0.00045623729771974796 0.00019025805491009606 0.00012887851763090902

    # ot_solver.plot( pot )

    # class S:
    #     def __init__( self ):
    #         R = numpy.random.random( [ 10, 10 ] )
    #         self.A = R.T @ R + numpy.eye( 10 ) * 2  # SPD
    #         self.B = numpy.random.random( [ 10 ] )
    #         self.sol = numpy.linalg.solve( self.A, self.B )

    #     def search_dir( self, pot ):
    #         # analogue de (measure - weight) / diag_Hessian  →  signe positif quand pot trop grand
    #         return ( self.A @ pot - self.B ) / numpy.diag( self.A )

    #     def system( self, pot, search_dirs ):
    #         # M[r,c] = s_r @ A @ s_c,  V[r] = s_r @ (A pot - B)   (même convention que l'OT)
    #         S = numpy.array( search_dirs )
    #         gradient = self.A @ pot - self.B
    #         M = S @ self.A @ S.T
    #         V = S @ gradient
    #         return M, V

    # ot_solver = S()
    # for nb_to_keep in [ 1, 2, 3 ]:
    #     numpy.random.seed( 0 )
    #     ot_solver.__init__()
    #     pot = numpy.zeros( [ 10 ] )
    #     pot = gc_eigen( ot_solver, pot, nb_to_keep )
    #     print( f"gc_eigen(nb_to_keep={nb_to_keep}) erreur finale : {numpy.linalg.norm( pot - ot_solver.sol ):.2e}" )
    #     print()

    # numpy.random.seed( 0 )
    # ot_solver.__init__()
    # pot = numpy.zeros( [ 10 ] )
    # pot = gc( ot_solver, pot )
    # print( f"gc (PCG) erreur finale : {numpy.linalg.norm( pot - ot_solver.sol ):.2e}" )
