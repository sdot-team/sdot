from .BatchOf1dOtPlans import BatchOf1dOtPlans
from .OtPlan import OtPlan

class BatchOfOtPlans:
    """
    barycenters : Tensor[ batch_index, dirac_index, dim ]
    distances : Tensor[ batch_index ]
    potentials : Tensor[ batch_index, dirac_index ]
    """

    def __init__( self, distances, barycenters, potentials = None ):
        assert barycenters.ndim == 3
        assert distances.ndim == 1

        self.barycenters = barycenters
        self.distances = distances
        self.potentials = potentials

    def unbatch( self ):
        assert self.barycenters.shape[ 0 ] == 1
        assert self.distances.shape[ 0 ] == 1

        return OtPlan( self.distances[ 0 ], self.barycenters[ 0, :, : ] )

    def unidimensionnal_version( self ):
        assert  self.barycenters.shape[ 2 ] == 1

        return BatchOf1dOtPlans( self.distances, self.barycenters[ :, :, 0 ] )
