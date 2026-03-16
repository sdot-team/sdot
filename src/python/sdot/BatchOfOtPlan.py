from .OtPlan import OtPlan

class BatchOfOtPlan:
    """
    barycenters : Tensor[ batch_index, dirac_index, dim ]
    distances : Tensor[ batch_index ]
    """

    def __init__( self, distances, barycenters ):
        assert barycenters.ndim == 3
        assert distances.ndim == 1

        self.barycenters = barycenters
        self.distances = distances

    def unbatch( self ):
        assert self.barycenters.shape[ 0 ] == 1
        assert self.distances.shape[ 0 ] == 1

        return OtPlan( self.distances[ 0 ], self.barycenters[ 0, :, : ] )
