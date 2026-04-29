
class Dyn:
    """
    Marks an axis as dynamic: allocated at full capacity, actual size tracked separately.

    In TensorField:  Tensor( "nb_vertices", "dim", axes_info = { "dim" : Dyn() } )
    In Return:       Return( Tensor, nb_points = reservation, dim = 3, axes_info = { "dim" : Dyn() } ] )
    """

    def __init__( self, name: str, capacity = None, one_value_for_each = [] ):
        self.one_value_for_each = one_value_for_each
        self.capacity = capacity
        self.name = name
