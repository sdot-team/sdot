class Distribution:
    """
    """
    def batch_version( self ):
        raise RuntimeError( f"To be redefined for { type( self ) }" )

    @property
    def dim( self ) -> int:
        raise RuntimeError( f"To be redefined for { type( self ) }" )
