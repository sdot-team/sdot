# Performance

This page gives the measured cost of sdot: a single power diagram, a sequence of diagrams on the same diracs, a whole
Newton solve, CPU against GPU, and how both scale with the number of diracs and with the dimension. Every table is read
from the result files of [`bench/page.py`](https://github.com/sdot-team/sdot/tree/main/bench/page.py), one per machine:
the [Reproducing](#reproducing) section says how to add one.

## Setup

<BenchTable table="setup" />

Protocol:

* Times are the minimum over a few runs ( one for the long solves ). CPU threads are pinned, one per physical core, and
  the CPU kernels are compiled for the machine ( the wheels serve a precompiled level, at most a few percent slower ).
* A Newton solve starts cold, from the Voronoi diagram ( zero weights ), tree build included. It stops when
  `max |mass - target| / target <= 1e-6`. Steps, linear solvers and continuation are the library's defaults.
* GPU times are for a **jitted** call ( JAX ). An eager call costs about 0.1 s more.
* `mixed` ( the GPU default ) runs the float kernel and switches to the double kernel when float precision is no longer
  enough. The final accuracy is the double one.

## In short

<BenchTable table="newton" />

The image and gaussian targets include a width continuation ( 16 to 22 smoothing stages ), which explains why they cost
more than the Lebesgue cases.

## One power diagram

The cost of computing every cell's measure for fixed weights: the inner loop of every solver, in nanoseconds per dirac.
On the CPU, the same clouds go through CGAL ( `Regular_triangulation_2/3`, bulk insertion ), which then rebuilds and clips
every cell so that both sides give the same result. CGAL's 2D triangulation is sequential, its 3D one runs in parallel
( TBB ); the cells are rebuilt over the threads. They also go
through geogram ( Lévy's `exploragram`: Delaunay triangulation of the lifted seeds, then the cells restricted to a mesh of
the box ), whose cell measures match sdot's to 1e-10. geogram builds the cells in parallel, but its triangulation is
sequential in 2D and scales poorly in 3D ( about 1.7× on 8 cores ), and it is most of the cost.

<BenchTable table="diagram" />

On the GPU ( kernel time only ):

<BenchTable table="gpu_diagram" />

In float, the cell measures match the double kernel to a median relative error of about 1e-14 in 2D. The `mixed` mode
then finishes the solve in double.

### Prepared once, many diagrams

A Newton solve, a Lloyd step or a gradient flow at fixed positions computes many diagrams on the same diracs. sdot builds
its tree and its structures once, then each new diagram reuses them, and the neighbours found by the previous one. CGAL
has nothing to reuse: each new weight vector costs its whole chain again ( the CGAL columns of the table above ).

<BenchTable table="sequence" />

## Scaling with the number of diracs

The number of Newton iterations barely depends on `n`. The total cost therefore follows the cost of one diagram plus one
linear solve, which are both close to linear in `n`.

<BenchTable table="scaling" />

At 1e7 diracs in 3D, the float kernel can no longer resolve the cells ( each cell is about 4.6e-3 wide in a unit box ), so
the solve switches to the double kernel. On cards with slow FP64 this switch costs half of the total time.

GPU memory is about 900 bytes per dirac in 2D and 1300 in 3D at 1e6. A solve that does not fit raises a `MemoryError`
before it starts.

## Scaling with the dimension

<BenchTable table="dims" />

Newton is not affected by the dimension: the iteration count and the cost of the linear solves stay the same up to 6D.
The whole cost is in the cells. A Laguerre cell of a uniform cloud has a number of vertices that grows very fast with `d`
( about 6 in 2D, 25 in 3D, 120 in 4D, 670 in 5D, 3300 in 6D ), and the cell is cut, measured and triangulated vertex by
vertex. Each added dimension multiplies the cost of a diagram by about 20, then by about 40 from 5D to 6D.

In practice, 4D is comfortable up to a few 10⁵ diracs and 5D up to about 10⁵. 6D is limited to about 10⁴ diracs.
Beyond that, the cost comes from the geometry of the cells, not from the solver.

## Semi-discrete OT: sdot and geogram

geogram ( Bruno Lévy's [exploragram](https://github.com/BrunoLevy/exploragram), `OptimalTransportMap2d / 3d` ) solves the
same problem with a damped Newton method of the same family ( Kitagawa, Mérigot and Thibert ). Here it gets the same
clouds, the same box, the same stopping criterion and the same cores:

<BenchTable table="ot_geogram" />

How it was run:

* geogram's own Newton loop ( `optimize_full_Newton` ) was kept as it is, with one change: it stops on sdot's
  `max |mass - target| / target <= 1e-6` instead of its own criterion, a relative RMS error of 1 %. Its default use, a
  multilevel L-BFGS to 1 %, solves an easier problem and is not timed here.
* Its linear solver is the default one: a conjugate gradient with a Jacobi preconditioner, tolerance 1e-3, at most 1000
  iterations. It reaches the cap at every step: in 2D the linear solves take most of the time, in 3D the diagrams do.
  A tighter tolerance does not change the time or the number of steps.
* The box is integrated as a mesh of simplices. Its cells match sdot's to 1e-10 ( table [One power diagram](#one-power-diagram) ).
* On the 2D clustered cloud, geogram's step control stalls: after 100 steps the residual is still a few 1e-6.
  Lévy's multilevel scheme, run with a Newton solver, stops after the first level: interpolating the weights to the next
  level leaves cells empty, and exploragram's Newton stops on an empty cell. The `equal` clouds have coincident seeds,
  which geogram does not merge.

## Exact connectivity

sdot is not built for this: its cells are cut in floating point, and two cells may disagree about a vertex where several
seeds are almost on a sphere. `dual()` gives the regular triangulation restricted to the box, checked against qhull's on
random clouds. Where seeds are tied ( a grid, rings, seeds on a circle ) it gives one of the valid triangulations of the
tie.

In 2D, the connectivity is REPAIRED. Each cell is cut in floating point and only gives its list of neighbours. An edge is
kept when both of its seeds list it. When two kept edges cross ( the two diagonals of a square of a grid ), a single power
test removes one. The faces left are triangulated by one seed each. A triangle that its three cells propose is kept as it
is; a larger face is cut by the triangulation of its polygon that lies lowest on the lifted paraboloid. Each choice is made
once, so no two cells can disagree, and the floats only rank the choices. On random clouds almost every seed takes the
direct path, where its neighbours agree all around it. A tie of hidden seeds that leaves a face touching itself sends the
call to the certified path below.

In 3D each cell is cut in floating point, then CERTIFIED. A cell passes when no plane came within `1e-11` ( relative ) of
one of its vertices during the cut, and no vertex lies that close to a plane that is not its own. Its vertices are then
those of the exact cell. Only the doubtful cells are cut again with canonical predicates ( in double-double, with a
symbolic perturbation of the ties ). On random clouds, about one cell in ten thousand is cut again. On a grid, every cell
is. Double-double is not always enough: on a 2D grid of 10⁶ seeds nudged by 1e-14, a square is cocircular to 4e-35 and two
canonical cells take different diagonals ( the repair does not depend on such a precision ).

<BenchTable table="dual" />

A triangulation is what CGAL and geogram are made for. In 2D sdot is in the same range, on a grid as well ( 4.3 µs per seed
on one thread, against 16 µs with the certified cells ). In 3D the certification still costs about a cell's worth on top of
the cell itself ( table [One power diagram](#one-power-diagram) ). On a grid, where every cell is cut canonically, it is ten
times slower.

## Reproducing

The benchmarks run through errand, on any machine it has an environment for.
`bench/page.py` runs the suite of this page in the environments it is given ( the CPU jobs where there is no card, the
GPU jobs where the driver runs on one ) and writes `docs/guide/bench/<machine>.json`, which these tables read:

```bash
cd ~/Projects                                             # the errand root
python sdot-ffi/bench/page.py lmo-numpy lmo-jax           # CPU, then GPU: docs/guide/bench/lmo.json
python sdot-ffi/bench/page.py n22-jax --name n22          # another machine: one more column group
python sdot-ffi/bench/page.py lmo-numpy --quick           # a smoke run ( lmo-quick.json, not shown )
python sdot-ffi/bench/page.py xyz-numpy --threads=1,8,-1  # 1, 8 threads and all the cores ( default 1,-1 )
```

`--full` adds the long rows ( 6D, 1e7 on the card ), `--only diagram,cgal` reruns some jobs ( the others keep their last
run ). The CGAL and geogram columns run programs built on the machine: `page.py` first builds what is missing or out of date
( `bench/build_baselines.py`: CGAL's and geogram's sources cloned into `~/.cache/sdot`, the libraries in a micromamba
environment of their own; Linux ). A new machine needs only its errand environments; the docstring of `page.py` lists the
steps. Each bench can also be run alone, with every
option: see [`bench/README.md`](https://github.com/sdot-team/sdot/tree/main/bench/README.md).
