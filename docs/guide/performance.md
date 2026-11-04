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
for the 1e5 solves, 128 otherwise ). The diagrams are prepared ones, in double ( see [Power Diagrams](./performance-diagram) ).

* **Against geogram** ( the same Newton problem, the same cores ): 1e7 diracs in 3D take 1674 s with geogram, 89 s with
  sdot. On clustered clouds geogram stalls before the tolerance; sdot does not.
* **Against CGAL**: its 2D triangulation is sequential, and its 3D one stops gaining past 64 threads. A 3D sdot diagram
  gains 86× on 128 cores.
* **Linear in `n`**: the number of Newton iterations barely depends on `n`, each one costs a diagram and a linear solve,
  both close to linear.
* **Up to 6D**: the solver does not mind the dimension, the cells do. 4D is comfortable up to a few 10⁵ diracs, 6D up to
  about 10⁴.

The benchmarks are on four pages:

* [Semi-discrete OT](./performance-ot): the general case, diracs anywhere in any dimension, against geogram; over the
  threads, the number of diracs and the dimension;
* [1D Transport](./performance-1d) and [Diracs on a Grid](./performance-grid): structures with paths of their own, much
  faster ( below );
* [Power Diagrams](./performance-diagram): the diagram alone and its exact dual, against CGAL and geogram.

The special structures:

* **1D**: no Newton loop, an exact sort ( diracs ) or the merge of two images ( pixels of degree 0 or 1 ). In single
  precision, 256 transports of 1e6 diracs take 0.91 s on 128 cores, 0.39 s on a 2080 Ti; against POT on two images
  discretized for it, sdot reaches 1e-7 on $W_2$ 2 to 5 times faster than POT on as many cores
  ( [1D Transport](./performance-1d) ).
* **Diracs on a regular grid**: the cells known in advance up to the weights, a fixed stencil instead of a tree. A
  4096 × 4096 grid against an image of affine pixels is solved 17 times faster than the same diracs given as a cloud
  ( 3.2 s on 128 cores ), a 2048 × 2048 one in 0.35 s on a 2080 Ti and 0.072 s on an H100
  ( [Diracs on a Grid](./performance-grid) ).

Every chart is read from the result files of
[`bench/page.py`](https://github.com/sdot-team/sdot/tree/main/bench/page.py), one per machine. The machines, the
protocol and how to read the charts over the threads ( in CYCLES ) are in the [appendix](#appendix-setup-and-protocol),
each page ends with the tables behind its charts, and [Reproducing](#reproducing) says how to add a machine.

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
