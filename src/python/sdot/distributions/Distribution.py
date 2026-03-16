class Distribution:
    """
    """
    def batch_version( self, batch_size ):
        raise RuntimeError( f"To be redefined for { type( self ) }" )

    @property
    def batch_size( self ) -> int:
        return 1

    @property
    def dim( self ) -> int:
        raise RuntimeError( f"To be redefined for { type( self ) }" )
