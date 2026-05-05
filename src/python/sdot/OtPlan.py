from .aggregate import Tensor, Return, aggregate
# from .distributions.BatchOfDistributions import BatchOfDistributions
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs
from .distributions.Distribution import Distribution
from .PowerDiagram import PowerDiagram
from .driver import driver

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

    # def __init__( self, f: Distribution, g: Distribution, call_solve = True ):
    #     self.f = f
    #     self.g = g

    #     if call_solve:
    #         self.solve()

    # def unidimensional_version( self ):
    #     assert self.barycenters.shape[ 1 ] == 1
    #     return OtPlan1d( self.distances, self.barycenters[ :, 0 ], self.potentials, self.cuts )

    def solve( self ):
        f, g, update_cb = self._ordered_f_and_g()

        self.power_diagram = PowerDiagram( f.positions, f.weights )

    # def _ordered_f_and_g( self ) -> tuple[ SumOfWeightedDiracs, Distribution, callable ]:
    #     if isinstance( self.f, SumOfWeightedDiracs ):
    #         def nothing_to_do():
    #             pass
    #         return self.f, self.g, nothing_to_do

    #     raise NotImplementedError

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

    # res = OtPlan(
    #     power_diagram = power_diagram,
    #     barycenters = barycenters,
    #     potentials = power_diagram.weights,
    #     distance = distance,
    #     f = f,
    #     g = g,
    # )

    # if inverted:
    #     pass

    # return res



