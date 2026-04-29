# from .object_with_tensors._methods import object_with_tensors, TensorField, ListOfTensorFields, driver
# from .distributions.BatchOfDistributions import BatchOfDistributions
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs
from .distributions.Distribution import Distribution
from .PowerDiagram import PowerDiagram
from .OtPlan1d import OtPlan1d

# @object_with_tensors
class OtPlan:
    """
    """

    # # outputs
    # barycenters = TensorField( "nb_diracs", "dim" )
    # potentials = TensorField( "nb_diracs" )
    # distance = TensorField()

    # # inputs
    # f: Distribution
    # g: Distribution

    # # intermediate data
    # power_diagram: PowerDiagram

    # def __init__( self, f: Distribution, g: Distribution, call_solve = True ):
    #     self.f = f
    #     self.g = g

    #     if call_solve:
    #         self.solve()

    # def unidimensional_version( self ):
    #     assert self.barycenters.shape[ 1 ] == 1
    #     return OtPlan1d( self.distances, self.barycenters[ :, 0 ], self.potentials, self.cuts )

    # def solve( self ):
    #     f, g, update_cb = self._ordered_f_and_g()

    #     self.power_diagram = PowerDiagram( f.positions, f.weights )
    #     info( self.power_diagram.newton_dir( g ) )

    # def _ordered_f_and_g( self ) -> tuple[ SumOfWeightedDiracs, Distribution, callable ]:
    #     if isinstance( self.f, SumOfWeightedDiracs ):
    #         def nothing_to_do():
    #             pass
    #         return self.f, self.g, nothing_to_do

    #     raise NotImplementedError
