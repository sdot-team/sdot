Power diagram
=============

Power diagram (or Laguerre diagram) can be considerer as a generalization of Voronoi diagrams where each seeds has weights, allowing notably displacement of the cells.

In python the wrapper class is named `PowerDiagram`. It handles 
* virtual seeds, for periodicity handling,
* any number of dimension (>= 1),
* a wide variety of scalar types,
* large vectors (notably for out-of-core or MPI computations).

It is optionaly equiped with an underlying measure (Lebesgue by default), for methods like `cell_integrals`, `cell.boundary_integrals`...

Internally, it is optimized for multicore and SIMD/SIMT instruction sets.

Construction and visualization
-----------------------------

```python
from sdot import PowerDiagram

# we add some 2D seeds. By default, weights are equal to 1, leading to a voronoi diagram
pd = PowerDiagram( positions = np.random.random( [ 40, 2 ] ) )

# plot can take pyplot figure as input 
pd.plot( plt )
plt.show()
```

gives something like:

![Unbounded 2D PowerDiagram](pd_0.png)

By default, power diagrams are unbounded, but it is possible to add bounds using some affine functions.

```python
import matplotlib.pyplot as plt
from sdot import PowerDiagram
import numpy as np

# we center the points around 0 for this example
pd = PowerDiagram( np.random.random( [ 40, 2 ] ) - 0.5 )

# Here we use a setter to set a value after the call of the constructor (we can do the same thing for positions, weights, ...)
# In this example we add 5 affine functions defined by the directions [ np.cos( a ), np.sin( a ) ] and offsets equal to 1
# A point x is exterior when `dot( direction, x ) - offset > 0` 
pd.boundaries = [ [ np.cos( a ), np.sin( a ), 1 ] for a in np.linspace( 0, 2 * np.pi, 5, endpoint=False ) ]

pd.plot( plt )
plt.show()
```

gives something like:

![Bounded 2D PowerDiagram](pd_1.png)


In the next example, we add periodicity:

```python
from sdot import PowerDiagram, VtkOutput
import numpy as np

pd = PowerDiagram( np.random.random( [ 40, 2 ] ) )

# periodicity on the y-axis: we virtually repeat the seed with `[ 0, +1 ]` and `[ 0, -1 ]` translations.
# The "transformations" are affine and defined by a tuple ( M, V ). A seed at position `x` becomes a virtual seed at position `M @ x + V`
pd.virtual_seed_transformations = [
    ( np.eye( 2 ), [ 0, +1 ] ),
    ( np.eye( 2 ), [ 0, -1 ] ),
]

# VtkOutput is a convenience class to produce .vtk file that can be visualized for instance in paraview
vo = VtkOutput()
pd.plot( plt )
vo.save( "pd.vtk" )
```

TODO: image


Cells
-----

Cells can be scanned using the method `for_each_cell`. It calls the function given as argument with a `Cell` instance, which expected methods like `integral`, `boundaries`, ...

Most of theses methods have vectorized counterparts in the `PowerDiagram` class. For instance for `cell.integral(...)` there is a method `power_diagram.cell_integrals(...)`.

One important remark: by default, to avoid unnecessary memory consumption, cells are computed on the fly. If the user plans to call methods that lead to the same cells being scanned several times (the diagram not being modified between the calls), with a low memory pressure, it is possible to request caching with `power_diagram.use_cache = True`.

One another important comment: cell may be suported on space with lower dimensionnality. In this case, methods like `vertex_coords` or `vertex_refs` (list of cuts indices for each vertex) will return lists of arrays with size < dim.

It is illustrated in the following example 

```python
import matplotlib.pyplot as plt
from sdot import Cell

cell = Cell( ndim = 3 )

# we create a triangle, infinitely extruded
cell.cut( [ -1,  0, 0 ], 0 )
cell.cut( [  0, -1, 0 ], 0 )
cell.cut( [ +1, +1, 0 ], 1 )

# naturally, there's no 3D vertex...
print( cell.nb_vertices ) # => 0

# and this cell is sill unbounded (in 3D)
print( cell.bounded ) # => False

# It's because we're in 2D
print( cell.true_dimensionality ) # => 2

# "td" is the shortcut for "true dimensionality".
# Method with with prefix return the information for the subspace defined by `cell.base`
print( cell.nb_vertices_td ) # => 3 (the 3 vertices of the triangle)

# we can get coordinates to represent these points in 3D
print( cell.vertex_coords_td @ cell.base ) # => [[0. 0. 0.] [1. 0. 0.] [0. 1. 0.]]

# visualization will show the 2D content with thiner lines
cell.plot( plt )
plt.show()
```
