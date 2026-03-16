
class OtPlan1d:
    """
        barycenters : Tensor[ dirac_index ]
        distances : Tensor[]
    """

    def __init__( self, distances, barycenters ):
        assert barycenters.ndim == 1
        assert distances.ndim == 0

        self.barycenters = barycenters
        self.distances = distances

    def unidimensionnal_version( self ):
        return self
