from sdot_pytorch import sdot_w2
from matplotlib import pyplot
import torch
import numpy
import os

pyplot.style.use( 'dark_background' )

def solve_bfgs( loss, params, on_iter = None ):
    lbfgs = torch.optim.LBFGS( [ params ], history_size = 20, max_iter = 10, line_search_fn = "strong_wolfe" )

    def closure():
        lbfgs.zero_grad()
        objective = loss( params )
        objective.backward()
        return objective

    old_params = params.clone().detach()
    tol_param = 1e-7  # Stabilité de la solution
    tol_grad  = 1e-7  # Stabilité du gradient (proche du minimum)
    for i in range( 50 ):
        lbfgs.step( closure )

        if on_iter:
            on_iter( params, i )

        # --- CRITÈRES DE CONVERGENCE ---
        with torch.no_grad():
            grad_norm = torch.norm( params.grad )
            param_diff = torch.norm( params - old_params )

            print(f"Itération {i:02d} | Grad Norm: {grad_norm:.2e} | Param Diff: {param_diff:.2e}")

            # Test de sortie
            if grad_norm < tol_grad:
                print( "=> Convergence : Le gradient est quasiment nul." )
                break
            if param_diff < tol_param:
                print( "=> Convergence : La solution est stabilisée." )
                break

            old_params.copy_( params )


def loss( dirac_coords ):
    angles = torch.asarray( numpy.linspace( 0, numpy.pi, nb_angles, endpoint = False ), dtype = torch.float )
    proj_m = torch.row_stack( ( torch.cos( angles ), torch.sin( angles ) ) )
    dirac_xs = proj_m.T @ dirac_coords.T
    dirac_ws = torch.ones_like( dirac_xs )
    point_xs = torch.linspace( -1, 1, 30 ).repeat( ( nb_angles, 1 ) )
    point_ys = torch.sqrt( 1 - point_xs ** 2 )

    return torch.sum( sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys ) )

def display_iter( dirac_coords, num_iter ):
    pyplot.plot( dirac_coords[ :, 0 ].detach().numpy(), dirac_coords[ :, 1 ].detach().numpy(), "*" )
    pyplot.xlim( [ -1.1, 1.1 ] )
    pyplot.ylim( [ -1.1, 1.1 ] )
    pyplot.axis( 'equal' )

    pyplot.savefig( f"build/img_{ 1000 + num_iter }" )
    pyplot.clf()


# solve
dirac_coords = torch.rand( ( 5000, 2 ), requires_grad = True )
nb_angles = 5000

solve_bfgs( loss, dirac_coords, on_iter = display_iter )

# affichage
coords = dirac_coords.detach().numpy()
# print( coords )

# pyplot.plot( dirac_coords[ :, 0 ].detach().numpy(), dirac_coords[ :, 1 ].detach().numpy(), "*" )
# pyplot.show()

os.system( "convert build/img_*png build/anim.gif" )
