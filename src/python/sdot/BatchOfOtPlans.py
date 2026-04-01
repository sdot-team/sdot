from .BatchOf1dOtPlans import BatchOf1dOtPlans
from .OtPlan import OtPlan

__all__ = [ "BatchOfOtPlans" ]

class BatchOfOtPlans:
    """
    barycenters : Tensor[ batch_index, dirac_index, dim ]
    distances   : Tensor[ batch_index ]
    potentials  : Tensor[ batch_index, dirac_index ]
    cuts        : Tensor[ batch_index, dirac_index, 2 ]
    """

    def __init__( self, distances, barycenters, potentials, cuts ):
        assert barycenters.ndim == 3
        assert distances.ndim == 1

        self.barycenters = barycenters
        self.potentials = potentials
        self.distances = distances
        self.cuts = cuts

    def unbatch( self ):
        assert self.barycenters.shape[ 0 ] == 1
        assert self.potentials.shape[ 0 ] == 1
        assert self.distances.shape[ 0 ] == 1
        assert self.cuts.shape[ 0 ] == 1

        return OtPlan( self.distances[ 0 ], self.barycenters[ 0, :, : ], self.potentials[ 0, : ], self.cuts[ 0, :, : ] )

    def unidimensional_version( self ):
        assert  self.barycenters.shape[ 2 ] == 1

        return BatchOf1dOtPlans( self.distances, self.barycenters[ :, :, 0 ], self.potentials, self.cuts )
