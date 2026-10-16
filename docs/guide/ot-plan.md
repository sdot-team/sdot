# W2, Plans and Settings

Three functions, from the most common to the most specific:

| | returns | when |
|---|---|---|
| `sdot.w2( f, g )` | $W_2^2$, a scalar | a distance, a loss to differentiate |
| `sdot.plan( f, g )` | the transport plan | barycenters, cell masses, weights, a picture |
| `sdot.ot_solve( f, g, ... )` | the transport plan | the same, with the settings of the solver |

`f` is the discrete source ( a `SumOfDiracs` ), `g` the continuous target ( an `Image`, a `SumOfGaussians`... see
[Distributions](./distributions/) ). The cost is the squared Euclidean distance.

## `w2`: the distance

<<< @/snippets/hello_numpy.py

The result is an array of the framework of the inputs ( numpy here, see [Frameworks and devices](./backends) ), and it is
differentiable with respect to everything that defines `f` and `g`.

## `plan`: what the transport does

<<< @/snippets/plan.py

The plan sends the cell of each dirac, in the power diagram of `weights`, onto that dirac. The barycenters are what the
Lloyd algorithm moves the diracs to ( see the [Lloyd quantization](/examples/lloyd-quantization) example ).

`p.transport()` gives `( cost, barycenters, cell_masses )` at once, and `p.power_diagram()` the diagram itself, to inspect
its cells.

## `ot_solve`: the settings of the solver

`ot_solve` computes the same plan, and takes the settings of the solver as keywords:

<<< @/snippets/ot_solve.py

In 2D and more, the solver is a damped Newton method ( Kitagawa, Mérigot, Thibert ), and its settings are:

| setting | default | |
|---|---|---|
| `tol` | `1e-8` | stop when every cell mass is within `tol` of its target ( the masses are normalized, the target of a dirac is `1 / n` ) |
| `max_iter` | `100` | Newton steps, at most |
| `ot_plan` | `None` | a previous plan to start from |
| `precision` | `"auto"` | `"fp64"`, `"fp32"`, or `"mixed"` on a GPU ( `"auto"`: fp64 on a CPU, mixed on a GPU ) |
| `continuation` | `"auto"` | solve first for a smoothed density: `"auto"` ( when a cell starts empty ), `"always"`, `"never"` |
| `on_failure` | `"raise"` | `"raise"` an `OtNotConverged` ( which carries the plan reached ), or `"ignore"` and read `p.converged` |

`p.stats` tells how it went: `nb_iter`, `residual`, `status`...

In 1D the plan is exact ( a sort ): there is neither a tolerance nor iterations, and the only setting is
`with_barycenters`:

<<< @/snippets/distributions_1d.py

## `OtProblem`: solving again and again

In a loop ( a descent, a reconstruction, Lloyd ), keep the problem and change its inputs. Each solve starts from the
previous solution, which is what makes a nearby transport cost a few Newton steps:

<<< @/snippets/ot_problem.py

`pb.solve( sdot.Iterative( tol = ... ) )` takes the settings of `ot_solve` ( `sdot.Direct( ... )` in 1D ).
