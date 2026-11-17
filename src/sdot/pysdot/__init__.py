"""A COMPATIBILITY LAYER for the code written for pysdot ( https://github.com/sd-ot/pysdot ), on top of sdot alone.

    from sdot.pysdot import OptimalTransport, PowerDiagram          # instead of `from pysdot import ...`
    from sdot.pysdot.domain_types import ConvexPolyhedraAssembly    # instead of `from pysdot.domain_types import ...`
    from sdot.pysdot.radial_funcs import RadialFuncInBall           # instead of `from pysdot.radial_funcs import ...`

or, to leave the imports of a program unchanged, once before them:

    import sdot.pysdot; sdot.pysdot.install()                       # `import pysdot` now gives this module

Same classes, same methods, same conventions ( the weights, the CSR matrices, the integrals not normalized ), computed by sdot.
What differs is said in the docstrings: `OptimalTransport` runs sdot's solver ( its `linear_solver` is ignored, the weights come
back with another additive constant ), a few outputs are not available ( `image_integrals`, the radial functions other than `RadialFuncUnit`, `RadialFuncInBall` and `RadialFuncEntropy` ), and
`display_jupyter` opens sdot's viewer. `sdot_diagram()` ( `PowerDiagram` ) and `plan` ( `OptimalTransport` ) give the sdot
objects underneath, for what goes further.
"""
from .OptimalTransport import OptimalTransport, BadInitialGuess
from .PowerDiagram import PowerDiagram
from .TransportMap import TransportMap, find_transport_map
from .domain_types import ScaledImage, ConvexPolyhedraAssembly
from . import domain_types, radial_funcs, util


def install():
    """makes `import pysdot` ( and `pysdot.domain_types`, `pysdot.radial_funcs`, ... ) give this module, for the programs written for
    pysdot to run unchanged. Raises if the real pysdot is already imported."""
    import sys
    current = sys.modules.get( "pysdot" )
    if current is not None and current is not sys.modules[ __name__ ]:
        raise RuntimeError( "sdot.pysdot.install(): the real pysdot is already imported" )
    # ( the submodules by `sys.modules`: their attributes in this package are the classes of the same name )
    for name in ( "", ".domain_types", ".radial_funcs", ".util", ".OptimalTransport", ".PowerDiagram", ".TransportMap" ):
        sys.modules[ "pysdot" + name ] = sys.modules[ __name__ + name ]
