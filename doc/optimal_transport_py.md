Optimal transport
=================

Sdot specializes in semi-discrete optimal transport, meaning that one of the two measures must be a sum of diracs.

The only transport cost currently managed is based on the Euclidean norm (research is underway for tools based on other norms... but this is still a long way from production).

Sdot provides tools for manipulating transport plans, and in particular for finding optimal ones.

Under the hood, it's largely based on the power diagram tools, for which major optimization efforts have been made (execution speed, memory handling, ...).

A basic example
---------------

This is a very explicit example, where all the inputs are in their final expected type:

```python
from sdot import optimal_transport_plan, SumOfDiracs, IndicatorFunction, UnitSquare
import matplotlib.pyplot as plt
import numpy as np

# find an optimal plan to go from diracs to an UnitSquare
tp = optimal_transport_plan(
    source_measure = SumOfDiracs( np.random.random( [ 40, 2 ] )  ),
    target_measure = IndicatorFunction( UnitSquare() ),
    # stopping criterion
    goal_for_inf_norm_of_mass_ratio_error = 1e-4, # actually the default value
    # what to display during execution
    log_of_inf_norm_of_mass_ratio_error = True
)

# tp is of type SdTransportPlan.
tp.plot( plt )
plt.show()

# In this case, `tp.forward_map` (of type `D2GTransportMap`) will typically provide methods that give informations for each dirac
print( tp.forward_map.kantorovitch_potentials ) # vector that corresponds to the weights of the powerdiagram
print( tp.forward_map.brenier_potentials ) # discrete Brenier potentials for each dirac (convex if `tp` is optimal)
print( tp.forward_map.barycenters ) # barycenters of the cells for each dirac

# `tp.backward_map` typically provides functions of coordinates or coordinates list.
x = [ 0.5, 0.6 ] # a point where target_measure is defined
print( tp.backward_map.kantorovitch_potential( x ) ) # scalar value of the Kantorovitch_potential function at `x`
print( tp.backward_map.brenier_potential( x ) ) # scalar value of the Brenier potentials (nicely convex of tp is optimal) at `x`
print( tp.backward_map.dirac_index( x ) ) # index of the dirac for cell at `x`
print( tp.backward_map.barycenter( x ) ) # position of the dirac for cell at `x`

# Besides, `tp` gives directly access to the power diagram where the generic (not discrete) measure is stored as the `underlying_measure`
print( tp.power_diagram.integrals() ) # => should give 1/40 for each cell
print( tp.power_diagram.weights ) # => Kantorovitch potentials for each cell
print( tp.power_diagram.brenier ) # => Brenier potentials for each cell
print( tp.power_diagram.barycenters ) # => first moment for each cell
print( tp.power_diagram.inertia_matrices ) # => second moment for each cell
```

It gives something like (arrows goes from the dirac positions to the barycenter of the cells)

![Unbounded 2D PowerDiagram](pd_2000.png)

By the way, due to automatic conversion and default values (aside from the log stuff), the same example could have been written in a more concise way:

```python
from sdot import optimal_transport_plan
import numpy as np

tp = optimal_transport_plan( np.random.random( [ 40, 2 ] ) )
tp.plot() # if not specified, `plot` proposes a display context and calls the corresponding `show` method.
```

As a matter of facts, the first measure in the arguments (if not named) is the source one. 

Besides, if measures are not already of type `Distribution`,
  * coordinates like input (convertible to a `numpy.ndarray` with `shape.size == 2`) are transformed to a `SumOfDirac`
  * `SpaceSubset` instances (like `UnitSquare` which inherits from it) are transformed to `IndicatorFunction`

The standard distributions offered by the `sdot` package are in `sdot.distributions`. `SpaceSubset` objects are defined in `sdot.space_subsets`.

Partial transport
-----------------

...

Moreau-Yosida regularization
----------------------------

...


Direct use of SdTransportPlan
-----------------------------

Functions like `optimal_transport_plan`, `optimal_partial_transport_plan` and `optimal_moreau_yosida_transport_plan` are actually very thin wrapper around the `SdTransportPlan` class. They simply call the constructor and the method `adjust_potentials`.

If you plan to call the aforementionned methods several times on similar datasets, it can be very profitable to use `SdTransportPlan` directly. It provides the same level of usability, while allowing a wide number of optimizations in terms of execution speed and memory usage.

Here is a simple example

```python
from sdot import SdTransportPlan
import numpy as np

# define a transport plan to go from some diracs to an UnitSquare
# this transport plan is not yet optimal
tp = SdTransportPlan( np.random.random( [ 40, 2 ] ) )

# one can define or redefined arguments that can be set using ctor. For instance:
tp.log_of_inf_norm_of_mass_ratio_error = True

# solve (uses a Newton by default)
tp.adjust_potentials()

# of course one has access to the previously seen methods
tp.plot()

# now, it's possible to change only some parts of the inputs without having to redefine everythinf
tp.dirac_positions = tp.cell_barycenters

# Make the Transport Plan Optimal Again
tp.adjust_potentials()

# it was the first iteration of a quantization procedure
tp.plot()
```

Common generic densities
------------------------

### IndicatorFunction

`IndicatorFunction` take a `SpaceSubset` instance as argument. Currently there are 3 concrete classes that inherits from `SpaceSubset`:
* the ubiquitous `UnitSquare` (that actually can be translated, scaled and rotated),
* `PolyhedralConvexSubspace` which contains a list of affine functions. A point `x` is in the subspace if all the function values at `x` are negative,
* `PolyhedralContouredSubspace` which contains a mesh, not necessarily convex.

### ImageFunction

`ImageFunction` takes an array, a transformation matrix and an interpolation type ('constant', 'linear', ...).

### SymbolicFunction

`SymbolicFunction` takes a symbolic expression, for instance in a `str`, where `x` represents the position in the global space. Where `r` represents for distance to the corresponding seed.

### MulFunction, AddFunction

It is possible to multiply or add densities (with optional coeffifient) using `MulFunction` and `AddFunction`.









