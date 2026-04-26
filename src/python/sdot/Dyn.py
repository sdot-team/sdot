
class Dyn:
    """
    Marks an axis as dynamic: allocated at full capacity, actual size tracked separately.

    In TensorField:  TensorField( Dyn( "nb_vertices" ), "dim" )
    In Return:       Return( Tensor, [ Dyn( "nb_points", reservation ), dim ] )
    """

    def __init__( self, name: str, capacity = None, one_value_for_each = [] ):
        self.one_value_for_each = one_value_for_each
        self.capacity = capacity
        self.name = name
