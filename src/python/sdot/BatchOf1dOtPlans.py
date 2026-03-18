from .OtPlan1d import OtPlan1d

__all__ = [ "BatchOf1dOtPlans" ]

class BatchOf1dOtPlans:
    """
        barycenters : Tensor[ batch_index, dirac_index ]
        potentials : Tensor[ batch_index, dirac_index ]
        distances : Tensor[ batch_index ]
        cuts : Tensor[ batch_index, dirac_index, 2 ]
    """

    def __init__( self, distances, barycenters, potentials, cuts ):
        assert barycenters.ndim == 2
        assert potentials.ndim == 2
        assert distances.ndim == 1
        assert cuts.ndim == 3

        assert cuts.shape[ 2 ] == 2

        self.barycenters = barycenters
        self.potentials = potentials
        self.distances = distances
        self.cuts = cuts

    def unbatch( self ):
        assert self.barycenters.shape[ 0 ] == 1
        assert self.potentials.shape[ 0 ] == 1
        assert self.distances.shape[ 0 ] == 1
        assert self.cuts.shape[ 0 ] == 1

        return OtPlan1d( self.distances[ 0 ], self.barycenters[ 0, : ], self.potentials[ 0, : ], self.cuts[ 0, :, : ] )

    def unidimensionnal_version( self ):
        return self
