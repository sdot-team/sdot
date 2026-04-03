from ..driver import driver
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
                v = getattr( self, a_name )
                if v is not None:
                    for item in v:
                        res.append( item )
                else:
                    for _ in range( getattr( self, a_data.main_axis_name ) ):
                        res.append( driver.empty( [] ) )
            if isinstance( a_data, TensorField ):
                v = getattr( self, a_name )
                if v is not None:
                    res.append( v )
                else:
                    res.append( driver.empty( [] ) )
        return res

    def unflat_tensor_list( self, out: list, inp: list ):
        from .helpers.distribution_methods import _collect_attributes, TensorField, ListOfTensorFields

        for _, a_data in _collect_attributes( type( self ) ):
            if isinstance( a_data, ListOfTensorFields ):
                loc = []
                for _ in range( getattr( self, a_data.main_axis_name ) ):
                    loc.append( inp.pop( 0 ) )
                out.append( loc )
            if isinstance( a_data, TensorField ):
                out.append( inp.pop( 0 ) )


def unflatten_args( f, g, args ):
    """ Convert a flat list of tensors to the structured list expected by the C++ binding. """
    res = []
    inp = list( args )
    f.unflat_tensor_list( res, inp )
    g.unflat_tensor_list( res, inp )
    return res
