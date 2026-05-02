from .aggregate import aggregate, Tensor
from typing import TYPE_CHECKING


@aggregate
class CutWorkspace:
    """

    """

    indices_to_remove : Tensor( "nb_indices_to_remove[]", dtype = int )
    used_flags : Tensor( "reservation[]", dtype = int )
    corr : Tensor( "reservation[]", dtype = int )
    sps : Tensor( "reservation[]" )

    map_items : Tensor( "nb_map_items[]", dtype = int )
    links : Tensor( "nb_links[]", dtype = int )

    if TYPE_CHECKING:
        def __default_init__( self, *args, **kwargs ): ...
        max_of_nb_indices_to_remove: int
        max_of_nb_map_items: int
        max_of_reservation: int
        max_of_nb_links: int


BatchOfCutWorkspace = CutWorkspace.BatchVersion
