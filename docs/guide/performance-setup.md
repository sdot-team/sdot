# Performance: Summary and Protocol

In short, on a 128-core CPU node and on two GPUs ( a GeForce RTX 2080 Ti and an H100 ):

| | CPU | H100 | 2080 Ti |
|---|---|---|---|
| OT solve, 1e5 diracs, 2D | 0.54 s | 67 ms | 0.11 s |
| OT solve, 1e5 diracs, 3D | 0.77 s | 0.11 s | 0.36 s |
| OT solve, 1e7 diracs, 2D | 205 s | 2.4 s | does not fit |
| OT solve, 1e7 diracs, 3D | 97 s | 16 s | does not fit |
| OT solve, 2048 × 2048 diracs on a grid, to an image | 1.1 s | 0.072 s | 0.35 s |
| one power diagram, 1e6 diracs, 2D / 3D, per dirac | 29 / 108 ns | 6 / 90 ns | 57 / 590 ns |
| one power diagram, 1024 × 1024 diracs on a grid, per dirac | 15 ns | | 35 ns ( as a cloud ) |

Solves start cold and stop at `max |mass - target| / target <= 1e-6`. The CPU column takes its best thread count ( 64
for the 1e5 solves, 128 otherwise ). The diagrams are prepared ones, in double ( see [Power Diagrams](./performance-diagram) ).

* **Against geogram** ( the same Newton problem, the same cores ): 1e7 diracs in 3D take 1674 s with geogram, 97 s with
  sdot. On clustered clouds geogram stalls before the tolerance; sdot does not.
* **Against CGAL**: its 2D triangulation is sequential, and its 3D one stops gaining past 64 threads. A 3D sdot diagram
  gains 85× on 128 cores.
* **Linear in `n`**: the number of Newton iterations barely depends on `n`, each one costs a diagram and a linear solve,
  both close to linear.
* **Up to 6D**: the solver does not mind the dimension, the cells do. 4D is comfortable up to a few 10⁵ diracs, 6D up to
  about 10⁴.
* **Special structures**: diracs on a regular grid have a solver of their own, a fixed stencil instead of a tree, 8 to 16
  times faster than the same diracs as a cloud ( [Semi-discrete OT](./performance#diracs-on-a-regular-grid) ); 1D plans
  need no Newton loop at all ( [1D Transport](./performance-1d) ).

The benchmarks are on three pages:

* [Semi-discrete OT](./performance): whole solves, against geogram; over the threads, the number of diracs and the
  dimension; diracs on a grid;
* [Power Diagrams](./performance-diagram): the diagram alone and its exact dual, against CGAL and geogram, clouds and
  grids;
* [1D Transport](./performance-1d): batches of 1D plans, against POT.

Every chart is read from the result files of
[`bench/page.py`](https://github.com/sdot-team/sdot/tree/main/bench/page.py), one per machine. Each page ends with the
tables behind its charts.

## Setup and protocol

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

## Reproducing

The benchmarks run through errand, on any machine it has an environment for.
`bench/page.py` runs the suite in the environments it is given ( the CPU jobs where there is no card, the GPU jobs where
the driver runs on one ) and writes `docs/guide/bench/<machine>.json`, which the charts and tables read:

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
