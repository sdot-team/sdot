from sdot_pytorch import sdot_w2
from matplotlib import pyplot
import torch
import numpy
pyplot.style.use( 'dark_background' )

def solve_bfgs( loss, params, on_iter = None ):
    lbfgs = torch.optim.LBFGS( [ params ], history_size=10, max_iter = 4, line_search_fn = "strong_wolfe" )

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

    # Clean up optimizer state
    del lbfgs
    del old_params


def loss( dirac_coords ):
    angles = torch.asarray( numpy.linspace( 0, numpy.pi, nb_angles, endpoint = False ), dtype = torch.float )
    proj_m = torch.row_stack( ( torch.cos( angles ), torch.sin( angles ) ) )
    dirac_xs = proj_m.T @ dirac_coords.T
    dirac_ws = torch.ones_like( dirac_xs )
    point_xs = torch.linspace( -1, 1, 30 ).repeat( ( nb_angles, 1 ) )
    point_ys = torch.sqrt( 1 - point_xs ** 2 )

    return torch.sum( sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys ) )

def display_iter( dirac_coords, num_iter ):
    # pyplot.plot( dirac_coords[ :, 0 ].detach().numpy(), dirac_coords[ :, 1 ].detach().numpy(), "*" )
    # pyplot.savefig( f"img_{ 1000 + num_iter }" )
    pass


# solve
dirac_coords = torch.rand( ( 1000, 2 ), requires_grad = True )
nb_angles = 100

solve_bfgs( loss, dirac_coords, on_iter = display_iter )

# Detach from computation graph
dirac_coords.requires_grad = False
dirac_coords.grad = None

# # affichage
coords = dirac_coords.detach().numpy()
print( coords )

# Clean up to avoid PyTorch dispatcher errors on exit
del dirac_coords
del coords
import gc
gc.collect()

# pyplot.plot( dirac_coords[ :, 0 ], dirac_coords[ :, 1 ], "*" )
# pyplot.show()
# * frame #0: 0x000000010072d514 Python`unicodekeys_lookup_unicode + 124
#   frame #1: 0x000000010072d8e0 Python`_Py_dict_lookup + 332
#   frame #2: 0x000000010074e340 Python`_PyObject_GenericGetAttrWithDict + 328
#   frame #3: 0x0000000100748e70 Python`_Py_module_getattro_impl + 56
#   frame #4: 0x000000010074ed28 Python`_PyObject_GetMethod + 580
#   frame #5: 0x00000001006df9c4 Python`PyObject_VectorcallMethod + 80
#   frame #6: 0x000000010087c5ac Python`wait_for_thread_shutdown + 108
#   frame #7: 0x000000010087bcec Python`_Py_Finalize + 152
#   frame #8: 0x00000001008b07f0 Python`Py_RunMain + 428
#   frame #9: 0x00000001008b0dcc Python`pymain_main + 232
#   frame #10: 0x00000001008b0e68 Python`Py_BytesMain + 40
#   frame #11: 0x0000000192551d54 dyld`start + 7184
