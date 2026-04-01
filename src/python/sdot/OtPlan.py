from .OtPlan1d import OtPlan1d

__all__ = [ "OtPlan" ]

class OtPlan:
    """
        barycenters : Tensor[ dirac_index, dim ]
        potentials : Tensor[ dirac_index ]
        distances : Tensor[]
        cuts : Tensor[ dirac_index, 2 ]
    """

    def __init__( self, distances, barycenters, potentials, cuts ):
        assert barycenters.ndim == 2
        assert potentials.ndim == 1
        assert distances.ndim == 0
        assert cuts.ndim == 2

        self.barycenters = barycenters
        self.potentials = potentials
        self.distances = distances
        self.cuts = cuts

    def unidimensional_version( self ):
        assert self.barycenters.shape[ 1 ] == 1
        return OtPlan1d( self.distances, self.barycenters[ :, 0 ], self.potentials, self.cuts )
