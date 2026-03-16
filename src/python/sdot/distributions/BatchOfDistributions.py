
class BatchOfDistributions:
    """
    """
    @property
    def batch_size( self ) -> int:
        raise RuntimeError( f"To be redefined for { type( self ) }" )

    @property
    def dim( self ) -> int:
        raise RuntimeError( f"To be redefined for { type( self ) }" )

    @property
    def always_1d( self ) -> bool:
        return False
