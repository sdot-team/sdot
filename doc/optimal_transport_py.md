Optimal transport
=================

Sdot specializes in semi-discrete optimal transport, meaning that exactly one of the two measures must be a sum of diracs.

The only transport cost currently managed is based on the Euclidean norm (research is underway for tools based on other norms... but this is still a long way from production).

Sdot provides tools for manipulating transport plans, and in particular for finding optimal ones.

Under the hood, it's largely based on the power diagram tools, for which major optimization efforts have been made (execution speed, memory handling, ...).

A basic example
---------------

This is a very explicit example, where all the inputs are in their final expected type:

```python
from sdot import optimal_transport_plan, SumOfDiracs, IndicatorFunction, UnitBox
import matplotlib.pyplot as plt
import numpy as np

# find an optimal plan to go from diracs to an UnitBox
tp = optimal_transport_plan(
    source_measure = SumOfDiracs( np.random.random( [ 40, 2 ] )  ),
    target_measure = IndicatorFunction( UnitBox() ),
    # stopping criterion
    goal_for_infinite_norm_of_mass_ratio_error = 1e-4, # actually the default value
    # what to display during execution (verbosity_level is a way to force display)
    display_of_infinite_norm_of_mass_ratio_error = True
)

# tp is of type SdTransportPlan.
tp.plot( plt )
plt.show()

# In this case, `tp.forward_map` (of type `D2GTransportMap`) will typically provide methods that give informations for each dirac
print( tp.forward_map.kantorovitch_potentials ) # a vector that corresponds to the weights of the powerdiagram
print( tp.forward_map.brenier_potentials ) # discrete Brenier potentials for each dirac (convex if `tp` is optimal)
print( tp.forward_map.barycenters ) # barycenters of the cells for each dirac

# `tp.backward_map` typically provides functions of coordinates or coordinates list.
x = [ 0.5, 0.6 ] # a point where target_measure is defined
print( tp.backward_map.kantorovitch_potential( x ) ) # scalar value of the Kantorovitch_potential function at `x`
print( tp.backward_map.brenier_potential( x ) ) # scalar value of the Brenier potentials (nicely convex of tp is optimal) at `x`
print( tp.backward_map.dirac_index( x ) ) # index of the dirac for cell at `x`
print( tp.backward_map.barycenter( x ) ) # position of the dirac for cell at `x`

# Besides, `tp` gives directly access to the power diagram where the generic (not discrete) measure is stored as the `underlying_measure`
print( tp.power_diagram.integrals() ) # => should give a vector with 1/40 for each cell
print( tp.power_diagram.weights ) # => Kantorovitch potentials for each cell
print( tp.power_diagram.brenier ) # => Brenier potentials for each cell
print( tp.power_diagram.barycenters ) # => first moment for each cell
print( tp.power_diagram.inertia_matrices ) # => second moment for each cell
```

This code will give something like (arrows goes from the dirac positions to the barycenter of the cells)

By convention the duality is under the form \psi(y)-\phi_i \leq |y-X_i|^2. otherwise x\in Lag_i if |y-X_i|^2-\phi_i\leq |y-X_j|^2-\phi_j for all j 

![Unbounded 2D PowerDiagram](pd_2000.png)

Default values and conversions
------------------------------

The same example could have been written in a more concise way:

```python
from sdot import optimal_transport_plan
import numpy as np

# First specified measure is the source one.
# Coordinates like input (convertible to a `numpy.ndarray` with `shape.size == 2`) are transformed to a `SumOfDirac`.
# By default, the first unspecified measure is the UnitBox
tp = optimal_transport_plan( np.random.random( [ 40, 2 ] ) )

# if not specified, `plot` tries to find a display context, and calls the corresponding `show` method.
tp.plot()
```

Besides, `SpaceSubset` instances (like `UnitBox()`) are automatically transformed to an `IndicatorFunction` if a `Distribution` is expected.

Here are more information on distributions(distributions.md) and space subsets(space_subsets.md) readily available in `sdot`.

Partial transport
-----------------

TBC

Moreau-Yosida regularization
----------------------------

TBC

Direct use of SdTransportPlan
-----------------------------

Functions like `optimal_transport_plan`, `optimal_partial_transport_plan` and `optimal_moreau_yosida_transport_plan` are actually very thin wrapper around the `SdTransportPlan` class. They simply call the constructor and the method `adjust_potentials` before returning the created instance.

If you plan to work on several successive transport plans that share similar datasets, it can be very profitable to directly use instances of `SdTransportPlan`. It provides the same level of usability, while allowing a wide number of optimizations in terms of execution speed and memory usage (for instance, it may avoid to recompute the acceleration structures, ...).

Here is a simple example

```python
from sdot import SdTransportPlan
import numpy as np

# define a transport plan to go from some diracs to an UnitBox (arguments of `optimal_transport_plan` are presents in the same way than in `SdTransportPlan`)
# this transport plan is not yet optimal
tp = SdTransportPlan( np.random.random( [ 40, 2 ] ) )

# one can define or redefine ctor arguments. For instance:
tp.display_of_infinite_norm_of_mass_ratio_error = True

# solve (uses a Newton solver by default)
tp.adjust_potentials()

# of course one has access to the previously seen methods (the functions like `optimal_transport_plan` return a `SdTransportPlan`)
tp.plot()

# now, it's possible to change only some parts of the inputs without having to redefine (and recompute) everything
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
* the ubiquitous `UnitBox` (that actually can be translated, scaled and rotated),
* `PolyhedralConvexSubspace` which contains a list of affine functions. A point `x` is in the subspace if all the function values at `x` are negative,
* `PolyhedralContouredSubspace` which contains a mesh, not necessarily convex.

### ImageFunction

`ImageFunction` takes an array, a transformation matrix and an interpolation type ('constant', 'linear', ...).

### SymbolicFunction

`SymbolicFunction` takes a symbolic expression, for instance in a `str`, where `x` represents the position in the global space. Where `r` represents for distance to the corresponding seed.

### MulFunction, AddFunction

It is possible to multiply or add densities (with optional coeffifient) using `MulFunction` and `AddFunction`.









