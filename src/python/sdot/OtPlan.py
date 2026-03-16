from .OtPlan1d import OtPlan1d

class OtPlan:
    """
        barycenters : Tensor[ dirac_index, dim ]
        distances : Tensor[]
    """

    def __init__( self, distances, barycenters ):
        assert barycenters.ndim == 2
        assert distances.ndim == 0

        self.barycenters = barycenters
        self.distances = distances

    def unidimensionnal_version( self ):
        assert self.barycenters.shape[ 1 ] == 1
        return OtPlan1d( self.distances, self.barycenters[ :, 0 ] )
