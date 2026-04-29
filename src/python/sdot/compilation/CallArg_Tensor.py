# from sdot.util.collect_attributes import collect_attributes_inst, collect_attributes, Annotation
# from sdot.aggregate import UndefinedTensor, Dyn
# from sdot.util.index import index

# from .StructInfo import StructInfo
# from .FfiOutput import FfiOutput
# from .Workspace import Workspace
# from .FfiInput import FfiInput
# from .Return import Return
# from .Tensor import Tensor

# from typing import Self
# import weakref
from .CallArg import CallArg

class CallArg_Tensor( CallArg ):
    """
    """

    # def configure_as_input_tensor( self, python_value: any, mutable: dict[ int ] | None, fai, driver, axis_names, ct_axes: dict[ int ], valid = True, represents_a_dynamic_axis = False ) -> Self:
    def __init__( self, *, python_value = None, shape: list ):


