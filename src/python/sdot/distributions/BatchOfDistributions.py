from typing import Self

class BatchOfDistributions:
    """
    Base class for all batch distributions.

    Subclasses should be decorated with @generate_distribution_methods to
    automatically receive __init__, _nd_positions, dim, is_a_1d_version, and one
    property per TensorField axis name.
    """

    MultidimensionalVersion    : type
    UnidimensionalVersion      : type
    BaseVersion                : type
    batch_size                 : int
    dim                        : int

    def multidimensional_version( self, *_ ) -> Self: ...

    @property
    def is_an_unidimensional_verion( self ) -> bool:
        return self.__class__.__name__.endswith( "1d" )

    @property
    def is_a_1d_version( self ) -> bool:
        """ true if comes from a multidimensional version (meaning that we can call .multidimensionnal_version()) """
        return False

    def flat_tensor_list( self ) -> list:
        from .helpers.distribution_methods import _collect_attributes, TensorField, ListOfTensorFields
        
        res = []
        for a_name, a_data in _collect_attributes( type( self ) ):
            if isinstance( a_data, ListOfTensorFields ):
                for item in getattr( self, a_name ):
                    res.append( item )
            if isinstance( a_data, TensorField ):
                res.append( getattr( self, a_name ) )
        return res

    # def __getattr__( self, name: str ) -> int:
    #     # Never actually called for attributes that exist; signals to static
    #     # type checkers (PyLance, mypy) that dynamically generated axis-name
    #     # properties (nb_diracs, batch_size, …) are intentional and return int.
    #     raise AttributeError( name )
