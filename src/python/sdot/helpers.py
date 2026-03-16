import torch

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
