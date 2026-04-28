
class CtKnown:
    """
    Marks an axis as compile time known:

    In TensorField:  TensorField( "nb_vertices", CtKnown( "dim" ) )
    In Return:       Return( Tensor, [ "nb_points", CtKnown( "dim" ) ] )

    If specified, "limit" says from which value axis stops to be compile time known
    """

    def __init__( self, name: str, limit = None ):
        self.limit = limit
        self.name = name
