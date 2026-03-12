from matplotlib import pyplot
import numpy
import torch
import sys
import os

sys.path.append( os.path.join( os.getcwd(), 'python/pytorch' ) )
sys.path.append( os.path.join( os.getcwd(), 'build' ) )
from sdot_pytorch import sdot_w2


def loss( dirac_xs ):
    dirac_ws = torch.ones_like( dirac_xs )
    point_xs = torch.linspace( -1, 1, 3 )
    point_ys = torch.sqrt( 1 - point_xs ** 2 )

    return sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys )

dirac_xs = torch.linspace( 0.0, 1.0, 101, requires_grad = True )
lbfgs = torch.optim.LBFGS( [ dirac_xs ], history_size=10, max_iter = 4, line_search_fn = "strong_wolfe" )

def closure():
    lbfgs.zero_grad()

    objective = loss( dirac_xs )
    objective.backward()
    return objective

# boucle
old_params = dirac_xs.clone().detach()
tol_param = 1e-5  # Stabilité de la solution
tol_grad = 1e-4   # Stabilité du gradient (proche du minimum)
for i in range( 3 ):
    #history_lbfgs.append(f(x_lbfgs).item())
    lbfgs.step( closure )

    # --- CRITÈRES DE CONVERGENCE ---
    with torch.no_grad():
        grad_norm = torch.norm( dirac_xs.grad )
        param_diff = torch.norm( dirac_xs - old_params )

        print(f"Itération {i:02d} | Grad Norm: {grad_norm:.2e} | Param Diff: {param_diff:.2e}")

        # Test de sortie
        if grad_norm < tol_grad:
            print("=> Convergence : Le gradient est quasiment nul.")
            break
        if param_diff < tol_param:
            print("=> Convergence : La solution est stabilisée.")
            break

        old_params.copy_( dirac_xs )

# affichage
dirac_xs = dirac_xs.detach().numpy()
print( 'Final position:', dirac_xs )

pyplot.plot( dirac_xs, numpy.ones_like( dirac_xs ), "*" )
pyplot.show()
