<!-- written by scripts/api_docs.py from the docstrings: do not edit -->

# API — W2 and Plans

The transport: `w2` for the distance, `plan` for the plan, `ot_solve` for the plan with the settings of
the solver, `OtProblem` to solve again and again. See [Guide: plans and settings](/guide/ot-plan).

::: v-pre

## `w2`

```python
w2(f, g, periodicity=None)
```

`W_2^2` between `f` and `g` ( on the quotient space with a `periodicity` )

## `plan`

```python
plan(f, g, periodicity=None, radial_func=None)
```

The transport plan from `f` to `g`, with the default settings ( `ot_solve` takes them ). `periodicity`: on a torus,
a cylinder... `radial_func`: a function of the distance to each dirac, integrated by its cell ( see `OtProblem` )

## `ot_solve`

```python
ot_solve(source, target, *, periodicity=None, radial_func=None, verbose=False, **settings)
```

The transport plan from `source` to `target`, with the settings of the solver as keywords: those of `Iterative` in
2D and more ( `tol`, `max_iter`, `ot_plan`... ), of `Direct` in 1D. A name that does not belong to them raises.

    p = ot_solve( SumOfDiracs( positions ), image, tol = 1e-10 )
    p2 = ot_solve( SumOfDiracs( positions + 0.01 ), image, ot_plan = p )   # starts from p

To solve again and again, keep an `OtProblem`.

## `OtProblem`

```python
OtProblem(source, target, periodicity=None, radial_func=None)
```

A transport problem, to solve once or again and again.

    pb = OtProblem( SumOfDiracs( positions ), target )
    p = pb.solve()                                  # the plan
    pb.source = SumOfDiracs( new_positions )        # change an input...
    p = pb.solve()                                  # ... and solve again, from the previous solution

`source` is a sum of diracs, `target` a density, which also gives the domain of the transport. Each `solve` starts
from the last solution of this problem, which is what makes a nearby problem cost a few Newton steps: keep the
problem in a loop ( a descent, Lloyd, a reconstruction ) rather than calling `ot_solve` again.

| Attribute | |
|---|---|
| `domain` | The domain of the transport, as half-spaces `( directions, offsets )`: `directions . x <= offsets`. |
| `nb_dims` | The dimension of the space. |
| `nb_diracs` | The number of diracs of the source. |
| `periodicity` | the periodicity of the space ( `None`: none ), see `sdot.periodicity` |
| `radial_func` | the radial function around each dirac ( `None`: none ), see `sdot.RadialExp` |
| `regime` | `"iterative"` ( a Newton method, in 2D and more ) or `"direct"` ( a sort, in 1D ): it follows from the inputs. |
| `source` | the discrete distribution, normalized |
| `target` | the target density, normalized -- and what gives the domain |

### `.solve(settings=None, verbose=False)`

The plan: a `SdotPlanNd` ( or a `SdotPlan1d` in 1D ). `settings`: an `Iterative` ( a `Direct` in 1D ), the default
ones when `None`. Starts from the last solution of this problem unless `settings` gives a starting point.

## `Iterative`

```python
Iterative(tol=1e-08, max_iter=100, ot_plan=None, weights0=None, continuation='auto', precision='auto', aggregate=True, keep_weights=False, tuning=None, on_failure='raise', differentiable='auto', duplicates='split')
```

The settings of the solver in 2D and more ( a damped Newton method ), for `OtProblem.solve` -- `ot_solve` takes
the same names as keywords.

* `tol`: the solve stops when every cell mass is within `tol` of its target ( the masses are normalized: the target
  of a dirac is `1 / n` by default ).
* `max_iter`: the number of Newton steps, at most.
* `ot_plan`: a previous plan to start from -- a nearby problem then costs a few steps. `weights0`: starting
  weights instead, when there is no plan.
* `continuation`: solve first for a smoothed version of the density, then sharper and sharper ones -- what a density
  with narrow bumps or empty regions needs. `"auto"` ( when a cell starts empty ), `"always"`, `"never"`.
* `precision`: the float of the geometry, `"fp64"`, `"fp32"`, or `"mixed"` ( fp32 then fp64, on a GPU ). `"auto"`:
  fp64 on a CPU, mixed on a GPU.
* `aggregate`: merge the diracs too close for the doubles to separate their cells ( see `SdotPlanNd.clusters` ).
* `duplicates`: what the diracs at the same position ( or too close for the doubles ) get: `"split"`, a piece of their
  merged cell each, of their own mass ( `SdotPlanNd.groups` ), or `"merge"`, the merged cell shared by all of them.
* `keep_weights`: keep the weights of every step in `history`.
* `on_failure`: `"raise"` an `OtNotConverged`, or `"ignore"` and read `converged`.
* `differentiable`: attach the derivatives of the outputs -- `"auto"` when an input is traced ( `jax.grad`... ).
* `tuning`: the settings of the benchmarks -- unstable, not for use.

## `Direct`

```python
Direct(with_barycenters=False)
```

The settings of the 1D plan, for `OtProblem.solve` -- `ot_solve` takes the same names as keywords. The 1D plan is
exact ( a sort ): there is no tolerance. `with_barycenters` stores the barycenters, which are otherwise not kept.

## `OtNotConverged`

```python
OtNotConverged(solution)
```

Raised when the solver stops without reaching its tolerance. `solution` is the plan it got to ( its `stats` say why ),
usable all the same; `Iterative( on_failure = "ignore" )` returns it without raising.

## `SdotPlanNd`

```python
SdotPlanNd(src_dist, dst_dist, verbose=False, **kwargs)
```

A transport plan in 2D and more, from `plan`, `ot_solve` or `OtProblem.solve`.

Each dirac receives a cell of the power diagram of `weights`, whose mass under the target density is the mass of the
dirac: the plan sends each cell onto its dirac. `cost` is `W_2^2`, `barycenters` the barycenter of each cell,
`cell_masses` their masses. `converged` and `stats` say how the solve went.

| Attribute | |
|---|---|
| `barycenters` | the barycenter of each cell, `[ n, d ]` ( its dirac if the cell is empty ) -- with a periodicity, next to the dirac as it was given ( the cells are built around the folded diracs: the folding is undone here ) |
| `cell_masses` | the mass of each cell, `[ n ]` |
| `clusters` | Which cluster each dirac belongs to ( `[ n ]`, the smallest index of its cluster ) when diracs too close to be separated were merged, else `None`. |
| `converged` | Whether the tolerance was reached. |
| `cost` | the transport cost, `W_2^2` |
| `groups` | The groups of the plan ( `sdot.Groups`: seeds that share a cell and split it, `Iterative( duplicates = "split" )` ), else `None`. |
| `history` | one dict per Newton step ( the first is the starting point ): `t` the step length, `residual_l2`, `max_abs_residual`, `min_measure`, `nb_diag`... and `weights` with `Iterative( keep_weights = True )` |
| `problem` | the problem this is the solution of |
| `sphere_areas` | With a sink ( partial transport ): the measure of the sphere inside each cell, where the target is not zero -- a length in 2D, an area in 3D, `[ n ]` ( `PowerDiagram.sphere_areas` ). What a surface energy counts: differentiable with respect to the positions and the masses of the diracs when they are traced, through the weights ( the implicit function theorem ). |
| `stats` | how the solve went, a dict: `status`, `nb_iter`, `residual`, `nb_diag` ( the power diagrams computed ), timings... |
| `target_masses` | The mass each cell must reach ( `[ n ]` ): the masses of the diracs, normalized. |
| `weights` | the weights of the power diagram of the plan, `[ n ]` ( defined up to a constant: `weights[ 0 ] == 0` ) |

### `.add_to_viz(viz, weights=None, seeds=True, transport=False, target=True, **kwargs)`

Draws the plan into a `Visualizer`: the power diagram at the FITTED weights ( `weights`
to see another one ), its seeds, and -- when `transport` -- the segment from each seed to
the barycenter of its cell ( only with the fitted weights ). `target`: the target density
adds what it knows of itself ( `Distribution.add_to_viz` ). `kwargs` are those of
`PowerDiagram.add_to_viz`. For the quick ways to look at it -- `write_html`, `write_pvd`,
`show` -- see `Displayable`.

### `.cost_and_position_grad()`

`( cost, grad )`: `W_2^2` and its derivative with respect to the positions of the diracs, `2 m_i ( p_i - b_i )` ( `[ n, d ]` ).

### `.power_diagram(weights=None)`

The `PowerDiagram` of the plan, at its weights ( or at `weights` ): to inspect or draw its cells.

### `.residual(weights=None)`

The mass error of each cell, `[ n ]`, at the weights of the plan ( or at `weights` ). Costs one more power diagram.

### `.show(filename=None, title=None, height=500, **kwargs)`

In a notebook ( Jupyter, VS Code, Colab, ... ): displays the page inline, in an iframe
`height` pixels high -- the wheel and touch gestures are then left to the notebook, they
can be switched back on from the menu. Elsewhere, or when `filename` is given: writes the
HTML page ( `filename`, or a temporary file ) and opens it in the browser. Returns the
path written ( None when nothing was written ).

### `.to_viz(viz=None, title=None, **kwargs)`

A `Visualizer` holding this object ( `viz`, when given: it is then completed ). `title`
names a NEW scene; `kwargs` are those of `add_to_viz`.

### `.transport()`

`( cost, barycenters, masses )` at the FITTED weights: the cost `W_2^2 = sum_i int_{cell_i}
|x - p_i|^2 rho` ( a scalar `Tensor` ), the barycenter of each cell ( `[ n, d ]` ) and
its mass ( `[ n ]` ).

All three are OUTPUTS of the call -- the solver measures them at the fitted weights, in the
same C++ and with the same scratch. An empty cell keeps its seed as barycenter.

### `.write_html(filename, title=None, **kwargs)`

Writes the self-contained HTML page of this object. Returns the path written.

### `.write_pvd(filename, title=None, axes=(0, 1, 2), **kwargs)`

Writes this object for ParaView as a `.pvd` collection ( with its `.vtu` beside it ),
the entry point one opens. Returns the `.pvd` path.

### `.write_vtk(filename, title=None, axes=(0, 1, 2), **kwargs)`

Writes this object for ParaView, as a `.vtu`. Returns the path written.

## `SdotPlan1d`

```python
SdotPlan1d(src_dist, dst_dist, with_barycenters=False, auto_update=True)
```

THE SOLUTION of a semi-discrete transport in dimension 1: diracs on a line against a density.

It is exact and has no loop: the diracs are sorted, and the cumulative distribution function of the density is
inverted at the cumulated masses of the diracs. It is batched ( one plan per item of a batch of densities, e.g. per
angle of a sinogram ) and differentiable end to end. It is requested from an `OtProblem` ( `ot_solve( ... )`,
`plan( ... )` ), whose settings are a `Direct`.

`cost` is `W_2^2`; `barycenters` ( `[ n, 1 ]` ) is stored only with `Direct( with_barycenters = True )`.

## `quantization`

```python
quantization(target, positions, masses=None, *, periodicity=None, radial_func=None, target_mass=1.0, tol=0.02, max_iter=200, callback=None, verbose=False, ot_tol=None, seed=None, **settings)
```

Positions of diracs that quantize `target` optimally, by Lloyd iterations: each dirac moves to the barycenter of its
cell until the moves are small. Returns the positions, `[ n, d ]`.

* `positions`: the starting positions `[ n, d ]`, or a number of diracs, then drawn uniformly in the support of
  `target` ( with `seed` ).
* `masses`: the mass of each dirac `[ n ]`, equal by default.
* `periodicity`, `radial_func`: those of the transport ( see `OtProblem` ) -- Lloyd on a torus, a Klein bottle...; the
  positions then stay next to where they started, possibly outside the support.
* `target_mass`: the share of `target` the diracs take ( a partial transport, see `SumOfDiracs` ), `1` by default.
* `tol`: stop when the root mean square move is under `tol` times the size of the cells. `max_iter`: at most.
* `callback( iteration, positions, plan )`: called after each solve -- to record the iterations.
* `ot_tol` and the other keywords: the settings of the solver ( see `Iterative` ).

## `wrap`

```python
wrap(positions, periodicity, target=None)
```

The `positions` brought back onto the support of `target` by the group that `periodicity` describes ( onto the
parallelepiped of its translations anchored at the origin, without a target ).

:::
