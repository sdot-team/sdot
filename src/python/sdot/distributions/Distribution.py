class Distribution:
    """
    Base class for all (non-batch) distributions.

    Subclasses should be decorated with @generate_distribution_methods to
    automatically receive __init__, batch_version, _nd_positions, dim,
    always_1d, and one property per TensorField axis name.
    """

    @staticmethod
    def batch_class():
        raise RuntimeError( "To be redefined" )

    def batch_version( self, batch_size: int ):
        raise RuntimeError( f"To be redefined for { type( self ) }" )

    @property
    def batch_size( self ) -> int:
        return 1

    @property
    def dim( self ) -> int:
        raise RuntimeError( f"To be redefined for { type( self ) }" )

    def tensor_list( self ) -> list:
        raise RuntimeError( f"To be redefined for { type( self ) }" )

    # def __getattr__( self, name: str ) -> int:
    #     # Never actually called for attributes that exist; signals to static
    #     # type checkers (PyLance, mypy) that dynamically generated axis-name
    #     # properties (nb_diracs, nb_points, …) are intentional and return int.
    #     raise AttributeError( name )
