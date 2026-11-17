"""The first check of a wheel ( `[tool.loom.release] smoke` ), in a clean venv: a cell built and measured -- generation,
compilation with the headers the wheel ships, registration, execution."""
import numpy as np
from sdot import Cell

c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ] )
print( c.vertex_positions )
print( c.measure )
assert abs( float( np.asarray( c.measure ) ) - 2.0 ) < 1e-9, "measure != 2.0"
