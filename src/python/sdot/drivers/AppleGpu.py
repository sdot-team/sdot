from .Device import Device


class AppleGpu( Device ):
    @property
    def name( self ):
        return "metal"

    @property
    def signature( self ):
        return "metal"

    @property
    def cpp_type( self ):
        return "AppleGpu"

    @property
    def is_apple_gpu( self ):
        return True

    def nb_threads( self, **kwargs ):
        raise NotImplementedError( "Metal nb_threads not yet implemented" )

    def __repr__( self ) -> str:
        return "AppleGpu"

    def driver_version_for_jax( self, devices ):
        return devices( "METAL" )[ 0 ]
