from sdot.OtPlan1d import OtPlan1d

class BatchOf1dOtPlans:
    """
        barycenters : Tensor[ batch_index, dirac_index ]
        distances : Tensor[ batch_index ]
    """

    def __init__( self, distances, barycenters ):
        assert barycenters.ndim == 2
        assert distances.ndim == 1

        self.barycenters = barycenters
        self.distances = distances

    def unbatch( self ):
        assert self.barycenters.shape[ 0 ] == 1
        assert self.distances.shape[ 0 ] == 1

        return OtPlan1d( self.distances[ 0 ], self.barycenters[ 0, : ] )

    def unidimensionnal_version( self ):
        return self
