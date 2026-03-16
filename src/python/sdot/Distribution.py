class Distribution:
    @property
    def nb_diracs( self ) -> int:
        raise RuntimeError( f"Not applicable for { type( self ) }" )

    @property
    def batch_size( self ) -> int:
        raise RuntimeError( f"Not applicable for { type( self ) }" )
