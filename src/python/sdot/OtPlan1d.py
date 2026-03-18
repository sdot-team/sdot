
__all__ = [ "OtPlan1d" ]

class OtPlan1d:
    """
        barycenters : Tensor[ dirac_index ]
        potentials : Tensor[ dirac_index ]
        distances : Tensor[]
        cuts : Tensor[ dirac_index, 2 ]
    """

    def __init__( self, distances, barycenters, potentials, cuts ):
        assert barycenters.ndim == 1
        assert potentials.ndim == 1
        assert distances.ndim == 0
        assert cuts.ndim == 2

        assert cuts.shape[ 1 ] == 2

        self.barycenters = barycenters
        self.potentials = potentials
        self.distances = distances
        self.cuts = cuts

        self.distance = distances.item()

    def unidimensionnal_version( self ):
        return self
