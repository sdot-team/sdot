import numpy
import sdot
import sys

def gc_eigen( ot_solver, pot, nb_to_keep=2, threshold=1e-10, use_eigs = False ):
    kept_dirs = []  # directions A-orthogonales de l'itération précédente
    for _ in range( 10 ):
        z = ot_solver.search_dir( pot )
        ic( numpy.linalg.norm( z ) )

        # un seul appel system : [z] + kept_dirs
        M_gs, V_gs = ot_solver.system( pot, [ z ] + kept_dirs )

        poly = ot_solver.poly( pot, [ z ] + kept_dirs )
        # ic( poly.truncate( 2 ).argmin() )
        # ic( poly.argmin() )
        # # center plot on the quadratic minimizer
        # x_opt = numpy.linalg.solve( M_gs, V_gs )
        # half = numpy.abs( x_opt ).max() * 3
        # print( "x_opt =", x_opt, "  half =", half )

        # xs = numpy.linspace( -half, half, 40 )
        # ys = numpy.linspace( -half, half, 40 )
        # X, Y = numpy.meshgrid( xs, ys )
        # Z3 = numpy.array( [ [ poly[ [ x, y ] ]               for x in xs ] for y in ys ] )
        # Z2 = numpy.array( [ [ poly.truncate( 2 )[ [ x, y ] ] for x in xs ] for y in ys ] )

        # from matplotlib import pyplot
        # fig = pyplot.figure()
        # ax = fig.add_subplot( 111, projection='3d' )  # type: ignore[attr-defined]
        # ax.plot_surface( X, Y, Z3, alpha=0.7 )  # type: ignore[attr-defined]
        # ax.plot_surface( X, Y, Z2, alpha=0.5 )  # type: ignore[attr-defined]
        # ax.set_xlabel( 'dir 0' )  # type: ignore[attr-defined]
        # ax.set_ylabel( 'dir 1' )  # type: ignore[attr-defined]
        # pyplot.show()

        # update pot
        # X_gs = poly.truncate( 2 ).argmin()
        X_gs = poly.argmin()
        # X_gs = numpy.linalg.solve( M_gs, V_gs )
        for c, d in zip( X_gs, [ z ] + kept_dirs ):
            pot += c * d

        # A-orthogonalise z contre les directions gardées (Gram-Schmidt en A-norme)
        n_kept = len( kept_dirs )
        coeffs = [ M_gs[ 0, i + 1 ] / M_gs[ i + 1, i + 1 ] for i in range( n_kept ) ]
        for c, d in zip( coeffs, kept_dirs ):
            z -= c * d

        #
        kept_dirs.append( z )
        while len( kept_dirs ) > nb_to_keep:
            kept_dirs.pop( 0 )

        # 0.15815169659806913    0.15815169659806913
        # 0.05461503859992672    0.05544844063183461
        # 0.034739147923014795   0.037448601768125964
        # 0.009736285477986162   0.010349427032485451
        # 0.00588667152792209    0.006184825107198004
        # 0.0016232694072782724  0.0018500976693050606
        # 0.00020119457384064373 0.00022953719933548216
        # 6.6039467826527e-05    7.89335089482431e-05
        # 1.0900394859198868e-05 1.3276094800443742e-05
        # 5.259625155149015e-06  5.749520058640263e-06

        # reconstruit M et V pour search_dirs = kept_dirs + [z_orth] sans nouvel appel
        # n = n_kept + 1
        # M = numpy.zeros( ( n, n ) )
        # V = numpy.zeros( n )
        # M[ :n_kept, :n_kept ] = M_gs[ 1:, 1: ]                                          # bloc kept_dirs × kept_dirs
        # M[ n_kept, n_kept ]   = M_gs[ 0, 0 ] - sum( coeffs[ i ] * M_gs[ 0, i + 1 ] for i in range( n_kept ) )  # z_orth · A · z_orth
        # # blocs croisés kept_dirs × z_orth = 0 par construction
        # V[ :n_kept ] = V_gs[ 1: ]
        # V[ n_kept ]  = V_gs[ 0 ] - sum( coeffs[ i ] * V_gs[ i + 1 ] for i in range( n_kept ) )
        # ic( M )

        # ic| M: array([[ 2.04962923e-02, -1.04152284e-03,  1.49665529e-04, -4.14735279e-05,  0.00000000e+00],
        #               [-1.04152284e-03,  4.12034046e-03, -1.55775024e-04, 8.71883845e-06,  0.00000000e+00],
        #               [ 1.49665529e-04, -1.55775024e-04,  1.24573437e-03, -7.18382274e-06,  0.00000000e+00],
        #               [-4.14735279e-05,  8.71883845e-06, -7.18382274e-06, 7.68599168e-05,  0.00000000e+00],
        #               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  2.84917094e-05]])
        # search_dirs = kept_dirs + [ z ]

        # if use_eigs:
        #     # M est quasi-diagonale mais on diagonalise pour robustesse
        #     lam, Q = numpy.linalg.eigh( M )
        #     keep_mask = lam > threshold * lam[ -1 ]
        #     lam, Q = lam[ keep_mask ], Q[ :, keep_mask ]
        #     c       = Q.T @ V
        #     contrib = numpy.abs( c ) / numpy.sqrt( lam )
        #     r_orig  = Q @ ( c / lam )
        #     order   = numpy.argsort( -contrib )
        #     kept_dirs = [ Q[ :, idx ] @ numpy.array( search_dirs ) for idx in order[ :nb_to_keep ] ]
        # else:
        #     # M est diagonale par construction — pas besoin de eigh
        #     diag    = numpy.diag( M )
        #     keep_mask = diag > threshold * diag.max()
        #     r_orig  = numpy.where( keep_mask, V / diag, 0.0 )
        #     contrib = numpy.abs( r_orig )             # |step_i| dans chaque direction
        #     order   = numpy.argsort( -contrib )
        #     kept_dirs = [ search_dirs[ idx ] for idx in order[ :nb_to_keep ] if keep_mask[ idx ] ]

        # step = r_orig @ numpy.array( search_dirs )
        # pot  = pot - step

    return pot

def gc( ot_solver, pot ):
    p = None   # direction conjuguée courante
    zg = None  # z_prev · g_prev, pour beta

    for _ in range( 10 ):
        z = ot_solver.search_dir( pot )   # z = P⁻¹ g
        print( numpy.linalg.norm( z ) )

        # z · g via system — sert de numérateur pour alpha et beta
        _, V = ot_solver.system( pot, [ z ] )
        zg_new = V[ 0 ]

        # direction conjuguée : p = z + beta * p_prev
        if p is not None:
            beta = zg_new / zg
            p = z + beta * p
        else:
            p = z.copy()

        # pas optimal : alpha = (z·g) / (p·A·p)
        M, _ = ot_solver.system( pot, [ p ] )
        alpha = zg_new / M[ 0, 0 ]
        pot = pot - alpha * p

        zg = zg_new

    return pot

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
    gc_eigen( ot_solver, pot, nb_to_keep = 4 )

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
