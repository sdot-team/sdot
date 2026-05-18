from .aggregate import Tensor, Return, aggregate
# from .distributions.BatchOfDistributions import BatchOfDistributions
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs
from .distributions.Distribution import Distribution
from .PowerDiagram import PowerDiagram
from .drivers.driver import driver

@aggregate
class OtPlan:
    """
    """

    # tools
    power_diagram : PowerDiagram

    # outputs
    barycenters : Tensor( "nb_diracs", "dim" )
    potentials : Tensor( "nb_diracs" )
    distance : Tensor()

    # inputs
    f: Distribution # SumOfWeightedDiracs
    g: Distribution


    def plot( self ):
        self.power_diagram.plot( self.g )


def optimal_transport_plan( f: Distribution, g: Distribution, ground_metric = None ):
    # normalize f
    while True:
        n = f.normalized_version()
        if n == f:
            break
        f = n

    # normalize g
    while True:
        n = g.normalized_version()
        if n == g:
            break
        g = n

    #
    inverted = False
    if type( f ).BaseVersion != SumOfWeightedDiracs:
        inverted = True
        f, g = g, f

    weights = driver.zeros( f.positions.shape[ 0 ] )

    power_diagram = PowerDiagram( positions = f.positions, weights = weights, ground_metric = ground_metric )
    distance, barycenters = power_diagram.adjust_weights( dirac_masses = f.weights, target_distribution = g )

    res = OtPlan(
        power_diagram = power_diagram,
        barycenters = barycenters,
        potentials = power_diagram.weights,
        distance = distance,
        f = f,
        g = g,
    )

    if inverted:
        pass

    return res
