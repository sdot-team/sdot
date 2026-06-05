from .Device import Device
import os


class AppleGpu( Device ):
    """
    Apple GPU device (jax-metal).

    Option A: kernels run as CPU code on Metal's unified memory.
    CPU pointers returned by typed_data() are directly valid on Apple Silicon.
    """

    @property
    def name( self ):
        return "metal"

    @property
    def signature( self ):
        return "metal"

    @property
    def cpp_type( self ):
        return "ExecutionContext_Cpu"

    @property
    def mem_type( self ):
        return "MemorySpace_CpuRam"

    @property
    def is_apple_gpu( self ):
        return True

    def nb_threads( self, nb_local_bytes_per_thread = 0, **kwargs ):
        n = os.cpu_count() or 1
        if nb_local_bytes_per_thread > 0:
            try:
                total = os.sysconf( 'SC_PHYS_PAGES' ) * os.sysconf( 'SC_PAGE_SIZE' )
                n = min( n, total // nb_local_bytes_per_thread )
            except ( AttributeError, ValueError ):
                pass
        return max( 1, n )

    def __repr__( self ) -> str:
        return "AppleGpu"

    def driver_version_for_jax( self, devices ):
        return devices( "METAL" )[ 0 ]
