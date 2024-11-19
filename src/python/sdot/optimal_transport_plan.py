from .SdotSolver import SdotSolver

def optimal_transport_plan( source_distribution, target_distribution, norm = 2, mass_error_ratio = 1e-4 ):
    """ 
        optimal transport plan between source_distribution and target_distribution
    """
    os = SdotSolver( source_distribution, target_distribution, norm )
    os.mass_error_ratio = mass_error_ratio
    return os.adjust_potentials()
