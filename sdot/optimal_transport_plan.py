from .SdotPlan import SdotPlan

def optimal_transport_plan( source_measure, target_measure = None, norm = 2, stop_when = None, display = None, relaxation = 1 ):
    """ 
        optimal transport plan between source_distribution and target_distribution
    """
    sp = SdotPlan( source_measure, target_measure, norm, display = display )
    sp.default_relaxation = relaxation

    if stop_when and 'max_error_ratio' in stop_when: # TODO: integration in SdotSolver
        sp.max_error_target = stop_when[ 'max_error_ratio' ]

    sp.adjust_potentials()
    return sp
