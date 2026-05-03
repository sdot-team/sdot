from .aggregate import aggregate, Tensor
from typing import TYPE_CHECKING


@aggregate
class MatrixTerms:
    """

    """

    matrix_rows : Tensor( "nb_matrix_terms[]", dtype = int )
    matrix_cols : Tensor( "nb_matrix_terms[]", dtype = int )
    matrix_vals : Tensor( "nb_matrix_terms[]" )
    vector_vals : Tensor( "nb_vector_terms" )
    residual    : Tensor()

    if TYPE_CHECKING:
        max_of_nb_matrix_terms : int
        nb_vector_terms : int

