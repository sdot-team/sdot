class BatchOfDistributions:
    """
    Base class for all batch distributions.

    Subclasses should be decorated with @generate_distribution_methods to
    automatically receive __init__, _nd_positions, dim, is_a_1d_version, and one
    property per TensorField axis name.
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

    @property
    def is_a_1d_version( self ) -> bool:
        """ true if comes from a multidimensional version (meaning that we can call .multidimensionnal_version()) """
        return False

    def tensor_list( self ) -> list:
        raise RuntimeError( f"To be redefined for { type( self ) }" )

    # def __getattr__( self, name: str ) -> int:
    #     # Never actually called for attributes that exist; signals to static
    #     # type checkers (PyLance, mypy) that dynamically generated axis-name
    #     # properties (nb_diracs, batch_size, …) are intentional and return int.
    #     raise AttributeError( name )
