# Performance: Semi-discrete OT


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

## Scaling with the threads

A Newton solve alternates diagrams and linear solves. In 3D the diagrams take most of the time on one thread, and the
solve gains 15× on 64 threads ( 25× on the clustered cloud ); what is left is in the linear solves and the tree. In 2D the
cells are cheap and the linear solves take a larger share: on several threads they go through a multigrid, which is
limited by the memory bandwidth, and the solve gains about 6× ( on one thread, a sparse Cholesky is used instead ). Past
64 threads the second socket adds memory traffic between the sockets and the solve gets slower again; on this node the
time of the linear solves also varies by about 2× from one run to the next, whatever the number of threads. geogram gains
about 2.4× in 3D and nothing in 2D.

## Scaling with the number of diracs

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

## Scaling with the dimension

<BenchChart chart="dims" />

Newton is not affected by the dimension: the iteration count and the cost of the linear solves stay the same up to 6D
( the [table](#dimension) ). The whole cost is in the cells. A Laguerre cell of a uniform cloud has a number of vertices
that grows very fast with `d` ( about 6 in 2D, 25 in 3D, 120 in 4D, 670 in 5D, 3300 in 6D ), and the cell is cut, measured
and triangulated vertex by vertex. At 1e4 diracs, a diagram costs about 6 times more in 3D than in 2D, then each added
dimension multiplies it by 12 ( 4D ), 25 ( 5D ) and 45 ( 6D ).

In practice, 4D is comfortable up to a few 10⁵ diracs and 5D up to about 10⁵. 6D is limited to about 10⁴ diracs.
Beyond that, the cost comes from the geometry of the cells, not from the solver.

## The tables

### Newton

<BenchTable table="newton" />

<BenchTable table="scaling" />

### sdot and geogram

<BenchTable table="ot_geogram" />

### Dimension

<BenchTable table="dims" />


The machines and the protocol ( cold starts, repetitions, clocks ) are on the [overview](./performance#appendix-setup-and-protocol).
