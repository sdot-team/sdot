# Performance

In short, on a 128-core CPU node and on two GPUs ( a GeForce RTX 2080 Ti and an H100 ):

| | CPU | H100 | 2080 Ti |
|---|---|---|---|
| OT solve, 1e5 diracs, 2D | 0.52 s | 67 ms | 0.11 s |
| OT solve, 1e5 diracs, 3D | 0.75 s | 0.11 s | 0.36 s |
| OT solve, 1e7 diracs, 2D | 183 s | 2.4 s | does not fit |
| OT solve, 1e7 diracs, 3D | 89 s | 16 s | does not fit |
| one power diagram, 1e6 diracs, 2D / 3D, per dirac | 29 / 110 ns | 6 / 90 ns | 57 / 590 ns |

Solves start cold and stop at `max |mass - target| / target <= 1e-6`. The CPU column takes its best thread count ( 64
for the 1e5 solves, 128 otherwise ). The diagrams are prepared ones, in double ( see [below](#one-power-diagram) ).

* **Against geogram** ( the same Newton problem, the same cores ): 1e7 diracs in 3D take 1674 s with geogram, 89 s with
  sdot. On clustered clouds geogram stalls before the tolerance; sdot does not.
* **Against CGAL**: its 2D triangulation is sequential, and its 3D one stops gaining past 64 threads. A 3D sdot diagram
  gains 86× on 128 cores.
* **Linear in `n`**: the number of Newton iterations barely depends on `n`, each one costs a diagram and a linear solve,
  both close to linear.
* **Up to 6D**: the solver does not mind the dimension, the cells do. 4D is comfortable up to a few 10⁵ diracs, 6D up to
  about 10⁴.

Every chart is read from the result files of
[`bench/page.py`](https://github.com/sdot-team/sdot/tree/main/bench/page.py), one per machine. The machines, the
protocol and how to read the charts over the threads ( in CYCLES ) are in the [appendix](#appendix-setup-and-protocol),
with the [tables](#appendix-the-tables) behind the charts; [Reproducing](#reproducing) says how to add a machine.

## One power diagram

The cost of computing every cell's measure for fixed weights, the inner loop of every solver, per dirac. On the CPU, the
same clouds go through CGAL ( `Regular_triangulation_2/3`, bulk insertion ), which then rebuilds and clips every cell so
that both sides give the same result, and through geogram ( Lévy's `exploragram`: Delaunay triangulation of the lifted
seeds, then the cells restricted to a mesh of the box ), whose cell measures match sdot's to 1e-10.

sdot is timed twice. From scratch, like the others: the tree, the diagram and its first measures. Then PREPARED: a Newton
solve, a Lloyd step or a gradient flow at fixed positions computes many diagrams on the same diracs, and each new one
reuses the tree and the neighbours found by the previous one. CGAL and geogram have nothing to reuse: each new weight
vector costs their whole chain again. On the cards ( kernel time, double ), a diagram is a prepared one.

<BenchChart chart="diagram" />

On the 128 cores and a uniform cloud of 1e6 diracs, a 2D diagram costs 167 ns per dirac from scratch and 29 ns prepared;
a 3D one 332 ns and 110 ns. From scratch, the tree is two thirds of the 2D time on 128 cores: it gains 5× up to 32
threads, then slows down once the threads span both sockets. An H100 brings a prepared diagram down to 6 ns in 2D and 90 ns in 3D, and one from scratch to 8 ns and
92 ns: the card builds its tree in one call, for almost nothing. A 2080 Ti takes 57 ns and 590 ns prepared, 61 ns and
600 ns from scratch: its double units are few ( 13 ns and 270 ns in float ).
In float, the cell measures match the double kernel to a median relative error of about 1e-14 in 2D: the `mixed` mode
then finishes the solve in double.

### Scaling with the threads

A diagram is cut cell by cell, with no shared state. In 3D a prepared diagram gains 86× on 128 cores ( 114× in cycles ).
In 2D the cells are cheap and the memory bandwidth stops the gain past 96 cores, at 52× ( 67× in cycles ). On the
clustered 2D clouds the cells are uneven and the gain is lower ( 14× to 25× ). CGAL's 2D triangulation is sequential. Its
3D one runs in parallel ( TBB ) and gains 9× up to 64 threads, then slows down once the threads span both sockets: at
128 threads it is twice as slow as on one thread. On the clustered 3D clouds it stops gaining past 16 threads and then
does the same. geogram builds the cells in parallel, but its triangulation is sequential in 2D and gains about 3× in 3D,
and it is most of the cost.

### Scaling with the number of diracs

The CPU on all its cores, and each card, from 1e3 to 1e7 uniform diracs, in time per dirac: a flat curve is a cost linear
in `n`.

<BenchChart chart="n_diagram" />

On a card, a diagram of less than about 1e5 diracs does not fill the SMs, and below 1e4 the launches set its cost. Up
there, an H100 computes a prepared 2D diagram in 5 ns per dirac at 1e7 ( float ), and a 3D one in 72 ns; a 2080 Ti in
16 ns and 310 ns. From scratch, the tree adds 1 to 4 ns per dirac.

## Semi-discrete OT

Geogram ( Bruno Lévy's [exploragram](https://github.com/BrunoLevy/exploragram), `OptimalTransportMap2d / 3d` ) solves the
same problem with the damped Newton method of Kitagawa, Mérigot and Thibert ( KMT ). sdot is a damped Newton method too,
but not KMT. In short:

* a log phase: the first iterations write the Newton system and the merit of the step on `log( mass / target )`, so a
  nearly empty cell asks for a fraction of its gap only ( half the diagrams on the hard 2D cases ); once
  `max |mass - target| / target <= 2`, the solver switches to the plain difference, the true Newton, which converges
  quadratically;
* the step is not halved from 1 at each iteration: the trials restart from the last accepted step, and the step
  comes from the limits at which the cells along the direction would be crushed;
* continuation: when cells start empty, the target density is first smoothed, then sharpened stage by stage
  ( `continuation` ); the Lebesgue cases below do not need it;
* seeds too close for the weights to separate are merged, then split again.

Here it gets the same clouds, the same box, the same stopping criterion and the same cores:

<BenchChart chart="ot_geogram" />

On all the cores, sdot solves 1e5 uniform diracs in 0.73 s in 2D and 1.0 s in 3D; its best times are on one socket, 64
threads: 0.52 s and 0.75 s ( see below ). On the clustered 2D cloud, geogram does not reach the tolerance ( see below ). An H100 solves the 2D case in 67 ms, a
2080 Ti in 0.11 s; the 3D case in 0.11 s and 0.36 s. The image and gaussian targets of the [tables](#newton) include a width continuation ( 16 to 22
smoothing stages ), which explains why they cost more than the Lebesgue cases.

How geogram was run:

* geogram's own Newton loop ( `optimize_full_Newton` ) was kept as it is, with one change: it stops on sdot's
  `max |mass - target| / target <= 1e-6` instead of its own criterion, a relative RMS error of 1 %. Its default use, a
  multilevel L-BFGS to 1 %, solves an easier problem and is not timed here.
* Its linear solver is the default one: a conjugate gradient with a Jacobi preconditioner, tolerance 1e-3, at most 1000
  iterations. It reaches the cap at every step: in 2D the linear solves take most of the time, in 3D the diagrams do.
  A tighter tolerance does not change the time or the number of steps.
* The box is integrated as a mesh of simplices. Its cells match sdot's to 1e-10.
* On the 2D clustered cloud, geogram's step control stalls: it reaches the tolerance in 2 runs of 9 ( one per thread
  count ), and after 100 steps the residual of the others is between 4e-6 and 5e-3.
  Lévy's multilevel scheme, run with a Newton solver, stops after the first level: interpolating the weights to the next
  level leaves cells empty, and exploragram's Newton stops on an empty cell. The `equal` clouds have coincident seeds,
  which geogram does not merge.

### Scaling with the threads

A Newton solve alternates diagrams and linear solves. In 3D the diagrams take most of the time on one thread, and the
solve gains 15× on 64 threads ( 25× on the clustered cloud ); what is left is in the linear solves and the tree. In 2D the
cells are cheap and the linear solves take a larger share: on several threads they go through a multigrid, which is
limited by the memory bandwidth, and the solve gains about 6× ( on one thread, a sparse Cholesky is used instead ). Past
64 threads the second socket adds memory traffic between the sockets and the solve gets slower again; on this node the
time of the linear solves also varies by about 2× from one run to the next, whatever the number of threads. geogram gains
about 2.4× in 3D and nothing in 2D.

### Scaling with the number of diracs

<BenchChart chart="n_newton" />

The number of Newton iterations barely depends on `n`. The total cost therefore follows the cost of one diagram plus one
linear solve, which are both close to linear in `n`. On the CPU, 1e7 diracs take 183 s in 2D and 89 s in 3D on 128 cores;
in 2D the linear solves are two thirds of it ( a uniform cloud of 1e7 seeds has close pairs, solved as groups,
`Iterative( duplicates = "split" )` ). A solve of 1e7 diracs takes 2.4 s in 2D and 16 s in 3D on an H100 ( an eager
call, timed by its kernels: its wall time also counts python's loop ). It does not fit on a 2080 Ti ( 11 GB ).

geogram solves 1e7 diracs in 3D in 1674 s ( 6 steps ). In 2D its number of steps grows with `n` ( 7 at 1e5, 26 at 1e6 ),
as its conjugate gradient stops at the cap before solving the system; the 1e7 solve was stopped after 5 hours without
having converged, so the chart has no point there.

At 1e7 diracs in 3D, the float kernel can no longer resolve the cells ( each cell is about 4.6e-3 wide in a unit box ), so
the solve switches to the double kernel. On cards with slow FP64 this switch costs half of the total time. GPU memory is
about 900 bytes per dirac in 2D and 1300 in 3D at 1e6. A solve that does not fit raises a `MemoryError` before it starts.

### Scaling with the dimension

<BenchChart chart="dims" />

Newton is not affected by the dimension: the iteration count and the cost of the linear solves stay the same up to 6D
( the [table](#dimension) ). The whole cost is in the cells. A Laguerre cell of a uniform cloud has a number of vertices
that grows very fast with `d` ( about 6 in 2D, 25 in 3D, 120 in 4D, 670 in 5D, 3300 in 6D ), and the cell is cut, measured
and triangulated vertex by vertex. At 1e4 diracs, a diagram costs about 6 times more in 3D than in 2D, then each added
dimension multiplies it by 12 ( 4D ), 25 ( 5D ) and 45 ( 6D ).

In practice, 4D is comfortable up to a few 10⁵ diracs and 5D up to about 10⁵. 6D is limited to about 10⁴ diracs.
Beyond that, the cost comes from the geometry of the cells, not from the solver.

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

<BenchChart chart="dual" />

A triangulation is what CGAL and geogram are made for, and on one thread they stay ahead, in 3D especially: the
certification costs about a cell's worth on top of the cell itself. On the 128 cores sdot takes 160 ns per seed in 2D and
440 ns in 3D. On a grid, where every cell is cut canonically, it is ten times slower.

### Scaling with the threads

3D gains 45× on 128 cores ( 60× in cycles ), the clustered 3D clouds 28× and 44×. A uniform 2D cloud gains 11× and stops
there from 32 threads on: a 2D cell is too little work for its memory traffic. The clustered 2D clouds gain 5× and 9×.
CGAL's 2D triangulation is sequential: its time on one thread is drawn flat ( dashed ) over the threads.

## Reproducing

The benchmarks run through errand, on any machine it has an environment for.
`bench/page.py` runs the suite of this page in the environments it is given ( the CPU jobs where there is no card, the
GPU jobs where the driver runs on one ) and writes `docs/guide/bench/<machine>.json`, which these charts and tables read:

```bash
cd ~/Projects                                             # the errand root
python sdot-ffi/bench/page.py lmo-numpy lmo-jax           # CPU, then GPU: docs/guide/bench/lmo.json
python sdot-ffi/bench/page.py n22-jax --name n22          # another machine: one more column group
python sdot-ffi/bench/page.py lmo-numpy --quick           # a smoke run ( lmo-quick.json, not shown )
python sdot-ffi/bench/page.py xyz-numpy --threads=1,8,24     # explicit thread counts ( default 1,-1: all the cores )
```

The charts and tables take their CPU runs from one machine and their GPU runs from every card listed: `SOURCES`, in
`docs/.vitepress/theme/benchTables.js`, names their files.

`--full` adds the long rows ( 6D, 1e7 on the card ), `--only diagram,cgal` reruns some jobs ( the others keep their last
run ). The CGAL and geogram runs use programs built on the machine: `page.py` first builds what is missing or out of date
( `bench/build_baselines.py`: CGAL's and geogram's sources cloned into `~/.cache/sdot`, the libraries in a micromamba
environment of their own; Linux ). A new machine needs only its errand environments; the docstring of `page.py` lists the
steps. Each bench can also be run alone, with every
option: see [`bench/README.md`](https://github.com/sdot-team/sdot/tree/main/bench/README.md).

## Appendix: setup and protocol

The CPU curves come from a server node, two AMD EPYC 7763 with 64 cores each ( 128 cores, no SMT ), which lets the thread
counts go well beyond a workstation's. The GPU points put two cards side by side: a consumer one, a GeForce RTX 2080 Ti,
and a datacenter one, an H100.

<BenchTable table="setup" />

Protocol:

* Times are the minimum over a few runs ( one for the long solves ). CPU threads are pinned, one per physical core, and the
  CPU kernels are compiled for the machine ( the wheels serve a precompiled level, at most a few percent slower ).
* A Newton solve starts cold, from the Voronoi diagram ( zero weights ), tree build included. It stops when
  `max |mass - target| / target <= 1e-6`. Steps, linear solvers and continuation are the library's defaults.
* GPU times are for a **jitted** call ( JAX ). An eager call costs about 0.1 s more.
* `mixed` ( the GPU default ) runs the float kernel and switches to the double kernel when float precision is no longer
  enough. The final accuracy is the double one.

The charts over the threads plot CYCLES, the time times the clock measured during the run: a single busy core runs faster
than 128 busy ones ( turbo ), and a plot of times would count that as a loss of scaling. A straight line of slope -1 is
perfect scaling. The cards are placed at their number of SMs, which an SM, with its four 32-wide schedulers, makes roughly
comparable to a core with its vector units: a point below the CPU curves at the same x does more with each unit.

## Appendix: the tables

### One power diagram

<BenchTable table="diagram" />

On the GPU, each card in its own columns:

<BenchTable table="gpu_diagram" />

A sequence of diagrams on the same diracs: the preparation, then each next diagram.

<BenchTable table="sequence" />

### Newton

<BenchTable table="newton" />

<BenchTable table="scaling" />

### sdot and geogram

<BenchTable table="ot_geogram" />

### Dimension

<BenchTable table="dims" />

### Exact connectivity

<BenchTable table="dual" />
