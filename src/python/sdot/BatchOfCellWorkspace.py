from .aggregate import batch_variant_of
from .CellWorkspace import CellWorkspace


@batch_variant_of( CellWorkspace )
class BatchOfCellWorkspace:
    pass
