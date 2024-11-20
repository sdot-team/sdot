from .SdotSolver import SdotSolver

def optimal_transport_plan( source_distribution, target_distribution = None, norm = 2, mass_error_ratio = 1e-4, relaxation = 1 ):
    """ 
        optimal transport plan between source_distribution and target_distribution
    """
    sp = SdotSolver( source_distribution, target_distribution, norm )
    sp.default_relaxation = relaxation
    sp.mass_error_ratio = mass_error_ratio
    sp.adjust_potentials()
    return sp
