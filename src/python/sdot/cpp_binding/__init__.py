"""sdot.cpp_binding — generic C++ binding machinery.

Example::

    cpp_binding( "test", "my_include.h" )( Output( v ), 10 )

Flow:
* from generic python object
* conversion to "standard" ones (int, array, …)
* call the binding with "standard" python objects as arguments
* conversion (in C++) to C++ objects with the same name
* call a C++ function with the same name
* conversion of the result to "standard" python outputs

Env variables
* SDOT_FORCE_BUILD: "1" to force build
* SDOT_CACHE_DIR: where to place the generated sources and dylibs
"""

from sdot.drivers.compilation.Output  import Output  as Output
from sdot.drivers.compilation.Return  import Return  as Return
from sdot.drivers.compilation.Mutable import Mutable as Mutable
from sdot.drivers.compilation.Tensor  import Tensor  as Tensor
from .CppBinding import CppBinding


cpp_binding = CppBinding()
