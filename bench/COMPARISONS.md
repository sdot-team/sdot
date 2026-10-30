# Comparisons to run (plan, 2026-10-26)

What to compare sdot against, why, and how. Each block says what question it answers for a reader, the
competitor(s), the protocol, the metric, and what already exists in `bench/`. Items marked **(check)** are
facts not yet verified against the competitor's code or docs.

The comparisons sit on four layers. A reader picks sdot for a reason at one of them, so each layer gets
its own table or figure:

| layer | the reader's question | competitors |
|---|---|---|
| A. one power diagram | "is the geometry fast and robust?" | CGAL, Voro++, geogram (CPU), Basselin et al. (GPU) |
| B. one semi-discrete solve | "how fast do I get the weights, and does it always converge?" | KMT damped Newton (baseline), L-BFGS, stochastic (OTT-JAX), geogram |
| C. the formulation | "why not discretize the density and use Sinkhorn / an LP?" | POT, geomloss (KeOps), OTT-JAX |
| D. 1D / projected | "is it faster than sorting?" | POT sliced, torch/jax sort |

Plus E, a feature table (no timing), which is what JOSS and the docs need first.

## Common rules

* Same machine, alone (errand's `bench` is exclusive), versions pinned and printed; competitors built from
  source with `-O3 -march=native` where they are C++.
* Same stopping rule for every solver: `max |mass - target| / target <= 1e-6` measured **by sdot on the returned
  weights** (one check, whatever the solver reports about itself).
* Cold start (Voronoi), tree / triangulation build included. Report the CPU thread count, and give each
  competitor the threads it can use.
* Each competitor run with the settings its authors recommend. Before publishing, send the numbers to the
  authors (Lévy, the CGAL / geomloss / OTT maintainers) to check them: it guards against a straw-man
  comparison and it is also a way to make sdot known to them.
* Ground truth where possible (C and D): product densities on the box, whose quadratic-cost map is the
  product of the 1D maps, and 1D cases in closed form.

## A. One power diagram (geometry)

The cases are the existing ones (`cases.py`: uniform 2D/3D, lines, planes, Voronoi and equal-mass weights)
plus a robustness set.

Every comparison gives the user **the same result on both sides**, through one of three scenarios:

* **cells**: the measure of every cell, so the dual is rebuilt on the triangulation side (CGAL: every cell from its neighbours,
  clipped to the box);
* **adjacency**: the neighbours of every cell, so the triangulation is rebuilt on the cell side (sdot: `hessian_rows`, neighbours
  and facet measures; `cell_complex`, the whole complex, glued). For CGAL this is triangulation + adjacency;
* **sequence**: the positions prepared once, then K diagrams with weights that change (a Newton solve, a Lloyd step).
  sdot reuses its tree and its neighbour memory; CGAL rebuilds a triangulation per weight vector.

| id | competitor | what is compared | status |
|---|---|---|---|
| A1 | CGAL `Regular_triangulation_2/3` (EPICK) + cell rebuild + clip | the three scenarios above, ns/seed; the clip is on CGAL's side, by hand (favours CGAL, said in `cgal/power_2d.cpp`) | `cgal_vs_sdot.sh`, results below |
| A2 | Voro++ `container_poly` (radical tessellation, weight = r²) | 3D ns/seed. Same paradigm as sdot's CPU path (cell by cell cutting), used in materials science (Laguerre microstructures) | to write |
| A3 | geogram restricted power diagram (CPU) | 3D ns/seed, restricted to a box and to a tetrahedral mesh | to write |
| A4 | Basselin, Alonso, Ray, Sokolov, Lefebvre, Lévy, *Restricted Power Diagrams on the GPU* (CGF 2021) | GPU ns/seed, same card; they report x50 against CPU for restricted power diagrams | code availability **(check)** |
| A5 | robustness: CGAL exact predicates, geogram (SoS) | degenerate inputs: exact grids (8 cells at a vertex), cocircular/cospherical seeds, near-coincident seeds, weights at the ulp. Metrics: sum of measures - 1, valid cell complex (Euler characteristic, every facet seen twice), float vs double | `cell_complex` canonical predicates exist; the test set is to gather from the tests |

Expected message: sdot's cells are computed independently (no global triangulation), so they parallelize
and run on the GPU. A1/A2 show the CPU cost, A4 the GPU, A5 that robustness is not traded away.

## B. One semi-discrete solve (strategy)

This is the comparison that shows sdot's specific technique. Most of it is an **ablation inside sdot**: same
diagrams and machine, only the strategy changes, so it is fully controlled and cheap. `bench_newton` already
has the switches.

| id | variant | flags / competitor |
|---|---|---|
| B1 | KMT damped Newton, the reference algorithm (Kitagawa, Mérigot, Thibert 2019) | `--step=trials --residual=lin` (restart t = 1: `--restart-factor=1e9`) |
| B2 | + log residual and switch | `--residual=log --residual-switch=2` |
| B3 | + limits step (2D) | `--step=limits` |
| B4 | + duplicate aggregation | `--aggregate=yes` (vs `no`) on `lines_equal`, `planes_equal` |
| B5 | + width continuation for sharp densities | `--continuation=always` vs none on `gauss4` (sigma 0.01-0.05), `image`, `image_hole` |
| B6 | linear solvers | `--linear-solver=cholesky,amg,mg`, the card's multigrid |
| B7 | quasi-Newton L-BFGS on the same diagrams (Mérigot 2011 style) and gradient ascent | to write: a scipy L-BFGS over `pd.measures` |
| B8 | OTT-JAX `solve_semidiscrete` (stochastic, eps >= 0) | to write; time to reach the same mass residual, or the residual reached in a time budget |
| B9 | geogram / exploragram semi-discrete Newton (Lévy 2015, 3D, and the cosmology code of Lévy, Mohayaee, von Hausegger 2021) | class name and how to run it **(check)**; 3D uniform and clustered, 1e5 .. 1e7 |

Metrics: Newton iterations, diagrams, time split (`t_diag`, `t_lin`, ...), **convergence curves** (residual
vs time) and the **failure rate** on the hard cases (KMT stalls or needs many trials there). The figure for a
paper is residual vs time, one curve per variant, on 3 cases (easy, clustered, sharp density).

The headline against B9: Lévy et al. report ~1e7 particles "within a few hours" on a personal computer;
sdot's card does 3D 1e7 in 55 s eager (A6000, `performance.md`). Same problem first (a uniform target, the
same tolerance), then their cosmology cloud if available. The distributed solver of Lévy (JCP 2025, 1e8+
points on a cluster) is out of reach on one card (memory: ~1300 B per dirac in 3D), state it as such.

## C. Semi-discrete vs discretizing the density

The question ML and imaging users ask. For a continuous target, discretize it with m points and solve
discrete-discrete; compare **accuracy against time and memory**.

| id | method | library |
|---|---|---|
| C1 | exact LP / network simplex | POT `ot.emd` (m, n up to ~1e4-1e5) |
| C2 | Sinkhorn, dense and on GPU | POT, OTT-JAX |
| C3 | multiscale Sinkhorn, KeOps | geomloss `SamplesLoss( "sinkhorn", blur, scaling )` |
| C4 | stochastic semi-discrete | OTT-JAX (B8) |
| C5 | sdot | `w2` / `OtPlan`, CPU and GPU |

Metrics, on product densities (exact map known) and an image:

* error on W2² and on the map (cell barycentres vs exact map at the diracs), as a function of time: a Pareto
  front per method, n from 1e3 to 1e7;
* **the gradient** of W2² with respect to the dirac positions (what flows, Lloyd, generative models use): error
  and time, sdot vs geomloss, the de facto choice in ML;
* memory (Sinkhorn dense is O(nm), the multiscale and KeOps ones are not).

Expected message: at equal accuracy sdot is orders of magnitude faster once the target is a density (no
blur bias, no m), and its gradient is exact. Where sdot does not win (small n, point targets) say so.

## D. 1D and projected

| id | compared | notes |
|---|---|---|
| D1 | `ContinuousOtPlan1d` (exact tents) vs sort-based quantiles on samples (torch/jax `sort`) | accuracy at equal time; 1e7 x 600 in 0.62 s on the card |
| D2 | sliced W2: `sdot.projected1d` vs POT `sliced_wasserstein_distance` | directions x n grid, GPU |
| D3 | tomography: tents vs a discretized LP / Sinkhorn reconstruction | the tomography-tents page as the case |

## E. Feature table (JOSS, docs)

Columns: sdot, geogram, CGAL, Voro++, pysdot, POT, geomloss, OTT-JAX. Rows: semi-discrete exact, dimension
(2, 3, any d), densities (piecewise constant/affine image, mesh, codimension mesh, radial, gaussian mixture),
partial / unbalanced, periodic, ground metrics, derivatives / autodiff, backends (numpy, torch, jax, cupy), GPU,
jit, precompiled wheels. Each cell checked against the competitor's docs, with the version.

## Priority

1. **E + A1 + B1-B6 + C3/C5 on two cases**: enough for the JOSS paper and the docs' performance page (the
   commented "Comparison with other libraries" section of `docs/guide/performance.md`). A1 and B1-B6 only need
   runs on lmo.
2. **B9 (geogram) + A4 (GPU restricted power diagrams) + A5**: the method paper on the solver and the
   geometry; contact Lévy before.
3. **C in full + D**: the ML / imaging papers.

# Results

## 2026-10-26, lmo ( Xeon W-2145, 8 cores / 16 threads ), CPU, double

Runs: `ablation.sh lmo-numpy "--threads=8 --pin=no"` and `cgal_vs_sdot.sh`; logs and `result.yaml` under `runs/` ( `-o` ).
During the ablation one core was taken by another job ( a GPU experiment, one CPU thread ), hence `--pin=no`: the
ablation's absolute times are a little pessimistic, its ratios are what it is for. The diagrams ( A1 ) ran on the machine
alone.

### B: the Newton ablation ( seconds, wall; diagrams in parentheses )

Lebesgue on the unit square / cube, n = 1e5, stop at `max |a - nu| / nu <= 1e-6`, min of 3. Each column adds one piece
to the previous one. "stag." = the solver stopped without converging.

| case | KMT | + restart | + log residual | + limits ( 2D ) | + aggregation | linear solver, best |
|---|---|---|---|---|---|---|
| uniform 2D | 2.44 (8) | 2.51 (8) | 2.48 (8) | 2.44 (7) | 2.60 (7) | amg 1.70 ( cholesky 2.46 ) |
| lines, Voronoi | 14.7 (78) | 12.2 (57) | 8.2 (32) | 5.5 (19) | 5.5 (19) | cholesky 5.5 ( amg 7.1 ) |
| lines, equal | stag. | stag. | stag. | stag. | **14.2 (20)**, C++ 5.4 | cholesky 13.5 |
| uniform 3D | 4.93 (8) | 4.83 (8) | 4.04 (6) | - | 4.25 (6) | amg 3.15 ( cholesky 912 ) |
| planes, Voronoi | 20.9 (27) | 21.1 (27) | 14.7 (17) | - | 15.0 (17) | amg 11.0 |
| planes, equal | 20.9 (27) | 20.9 (27) | 14.7 (17) | - | 14.7 (17) | amg 11.0 |

DENSITIES ( 2D, seeds uniform; gauss4 n = 1e5, images 512² n = 2e4 ). **Without the width continuation, no variant
converges on any of the five cases** ( stagnation, linear solver failure, or 100 iterations ). With it:

| case | KMT | + restart | + log | + limits | + aggregation | old campaign |
|---|---|---|---|---|---|---|
| gauss4 σ = 0.05 | 47.8 (130) | 47.2 (122) | 46.7 (121) | 43.5 (107) | 44.3 | 39 ( KMT ) |
| gauss4 σ = 0.02 | 127.9 (407) | 106.2 (293) | 102.0 (292) | 91.0 (208) | 89.9 | 124 ( KMT ), 108 ( limits ) |
| gauss4 σ = 0.01 | 222.1 (674) | 168.6 (447) | 168.1 (446) | 148.8 (298) | 147.6 | 153 ( KMT ) |
| image | 13.2 (118) | 12.2 | 12.1 | 11.9 | 12.3 | 3.6 ( limits, box-blur path ) |
| image with a hole | 12.5 (123) | 13.0 | 13.5 | 12.6 | 13.0 | 3.9 ( id. ) |

What it says:

* On clustered clouds the log residual and the limits step divide the diagrams by 4 ( lines: 78 -> 19 ) and the time by
  2.7; in 3D the log residual alone gives 1.4x ( 27 -> 17 diagrams ).
* Duplicates ( `lines_equal` ): every variant stagnates until the aggregation, which makes it converge ( witness 3.9e-12 ).
* Sharp densities: the continuation is not an optimisation, it is what makes Newton converge at all. The other pieces
  then give 1.5x at σ = 0.01 ( 674 -> 298 diagrams ).
* Uniform clouds: nothing to gain, KMT is already at 6-8 diagrams.

### A1: one power diagram, CGAL against sdot ( ns per seed, the machine alone )

CGAL `main` 72352b00 ( 2026-10-23 ), EPICK, sequential, `-O3 -march=native`. sdot double, 1 thread and 8 pinned threads.
`cgal_vs_sdot.sh`; the three scenarios of section A. sdot's "from scratch" is the tree plus the output.

**CELLS** ( the measure of every cell ). CGAL: triangulation + the cells rebuilt from the neighbours and clipped ( 3D: by the
old campaign's cut ). sdot: `cells + tree`.

| case | n | CGAL | sdot 1 thr | sdot 8 thr |
|---|---|---|---|---|
| 2D uniform | 1e6 | 1 499 | 2 230 + 1 434 | 301 + 483 |
| 2D lines, Voronoi | 1e5 | 1 357 | 2 063 + 1 390 | 324 + 1 230 |
| 2D lines, equal | 1e5 | 1 099 | 9 259 + 1 760 | 1 256 + 1 590 |
| 3D uniform | 1e6 | 13 660 | 14 761 + 1 739 | 1 963 + 544 |
| 3D uniform | 1e5 | 13 331 | 13 785 + 1 670 | 1 910-2 765 + 1 550 |
| 3D planes, Voronoi | 1e5 | 13 335 | 14 004 + 1 750 | 2 004 + 1 460 |
| 3D planes, equal | 1e5 | 13 269 | 36 661 + 2 090 | 5 086 + 1 810 |

**ADJACENCY** ( the neighbours of every cell ). CGAL: triangulation + the tour of each vertex. sdot: `hessian_rows` ( the
neighbours and the facet measures ) + tree; the whole glued `cell_complex` at 1e5, 8 threads.

| case | n | CGAL | sdot 1 thr | sdot 8 thr | sdot `cell_complex`, 8 thr |
|---|---|---|---|---|---|
| 2D uniform | 1e6 | 1 104 | 3 758 + 1 432 | 517 + 509 | 27 305 ( 1e5 ) |
| 2D lines, Voronoi | 1e5 | 984 | 12 222 + 1 390 | 1 443 + 1 250 | |
| 2D lines, equal | 1e5 | 732 | 22 323 + 1 760 | 3 099 + 1 620 | |
| 3D uniform | 1e6 | 4 971 | 81 179 + 1 736 | 10 448 + 548 | |
| 3D uniform | 1e5 | 4 755 | 66 072 + 1 680 | 8 811 + 1 420 | 206 288 |
| 3D planes, Voronoi | 1e5 | 4 764 | 67 765 + 1 740 | 9 121 + 1 480 | |
| 3D planes, equal | 1e5 | 4 882 | 97 973 + 2 140 | 12 763 + 1 820 | |

**SEQUENCE** ( K = 10 diagrams, the weights going from `W / 10` to `W`: the file's for the `equal` cases, otherwise uniform in
`[ -0.3 h², 0.3 h² ]` ). CGAL rebuilds everything per diagram: its CELLS chain on the final weights. sdot ( `bench_sequence` ):
preparation once, then the mean cost of each next diagram.

| case | n | CGAL per diagram | sdot 1 thr: prepare / next | sdot 8 thr: prepare / next | 8 thr, mean of the 10 |
|---|---|---|---|---|---|
| 2D uniform, Laguerre | 1e6 | 1 177 | 1 882 / 2 622 | 564 / 371 | 425 |
| 2D lines, Voronoi -> Laguerre | 1e5 | 844 | 1 808 / 2 327 | 1 386 / 416 | 571 |
| 2D lines, equal | 1e5 | 1 101 | 1 852 / 7 310 | 1 472 / 1 058 | 1 182 |
| 3D uniform, Laguerre | 1e6 | 12 933 ( 22 900 hidden ) | 2 277 / 17 755 | 703 / 2 386 | 2 557 |
| 3D planes, Voronoi -> Laguerre | 1e5 | 10 767 ( 17 889 hidden ) | 2 157 / 17 067 | 1 724 / 2 432 | 2 706 |
| 3D planes, equal | 1e5 | 13 567 | 2 175 / 28 439 | 1 881 / 4 044 | 4 200 |

The sums of the measures are 1 to 1e-6 on both sides. With `wscale = 0.3` in 3D, CGAL hides 2-18 % of the seeds ( empty
cells, no clip to do ): its 3D Laguerre rows are optimistic; a smaller `wscale` is to try.

What it says:

* **Sequentially, CGAL wins every scenario today**: CELLS 2D 1.5 µs against 3.7, 3D on par ( 13.7 against 16.5 ); SEQUENCE
  2D 1.2 µs against 2.6 per further diagram, 3D 12.9 against 17.8. In 2D this is the SIMD regression: the old engine was
  faster than CGAL there ( 146 ns per seed on 8 threads, 2.1x under the current one ).
* **With 8 threads sdot wins the CELLS and SEQUENCE scenarios**: 2D SEQUENCE 3.2x ( 371 against 1 177 ns per diagram ), 3D
  5.4x ( 2 386 against 12 933 ), and the card is far beyond ( `performance.md`: 3D 1e6 at 118-457 ns per seed on an A6000 ).
* **ADJACENCY is sdot's weak point**: `hessian_rows` costs 1.7-5.5x the measures ( 3D 1e5: 66 µs per seed on one thread,
  14x CGAL's lower bound, still 1.9x slower than sequential CGAL on 8 threads ), and `cell_complex` is host-side gluing
  ( 27 µs per seed in 2D, 206 in 3D: 40-60x CGAL ). A hypothesis to check for `hessian_rows`: its neighbour capacity starts at
  16 and loom doubles it by running the call again, and a 3D cell has 15.5 neighbours on average.
* The preparation is cheap next to a diagram in 3D ( 0.3 of a diagram on 8 threads ), not in 2D at 1e5 ( 3x: the tree's fixed
  cost, ~130 ms ).

### To fix before these numbers are published

1. **CPU cells 1.5-2.5x slower than the old C++** at 8 pinned threads ( new/old: 2D uniform 2.06, lines 1.54-1.71,
   3D 1e5 1.75-2.54, planes 1.74-1.91; only 3D 1e6 is level, 1.04 ). In 2D, the SIMD of the old engine is missing: it is what
   made it faster than CGAL. **Diagnosed** ( 2026-10-26, below ): the catalogue serves x86-64 baseline kernels, where the 2D
   register kernel's 8-lane `permute` runs through memory.
2. **The connectivity outputs**: `hessian_rows` ( 1.7-5.5x the measures ) and `cell_complex` ( host-side, 40-60x CGAL ).
3. **2D `linear_solver = "auto"` picks Cholesky**: 1.70 s of linear solves against 0.69 for AMG on uniform 2D ( total
   2.58 -> 1.41 s ); on lines Cholesky is the better one ( 1.96 against 3.59 ): the rule needs the cloud, not the dimension.
   `performance.md` says 1.02 s for uniform 2D: with `auto` it is 2.58 now.
4. **The aggregation costs ~9 s outside the C++** on `lines_equal` ( C++ total 5.4 s, wall 14.2 s ) for one cluster of
   2 seeds.
5. **Images with continuation 3.3x slower than the old campaign** ( 12 s against 3.6 ); the old path was a box-blur with
   an adaptive scale, not the same continuation: to compare before calling it a regression.
6. The tree's fixed cost at 1e5 ( ~130 ms ), and a Laguerre diagram 15-25 % dearer than a Voronoi one in SEQUENCE
   ( 2D 1e6, 8 threads: 371 against 301 ns ).

### The 2D SIMD regression ( diagnosed 2026-10-26 )

`bench_diagram --kernels=compile --target=...` ( new switches: compile for this machine, run as a level ), lmo, double, ns per seed:

| level | 2D uniform 1e6, 1 thr | 8 thr | lines Voronoi, 1 thr | 8 thr | lines equal, 1 thr | 8 thr |
|---|---|---|---|---|---|---|
| catalogue ( served as x86-64 ) | 2 281 | 408 | 2 094 | 420 | 9 484 | 1 676 |
| x86-64 ( SSE2 ) | 2 423 | 509 | 2 283 | 511 | 8 861 | 1 698 |
| x86-64-v2 | 2 497 | 497 | 2 346 | 509 | 9 543 | 1 922 |
| x86-64-v3 ( AVX2 ) | 2 285 | 450 | 2 178 | 483 | 7 713 | 1 422 |
| native = v4 ( AVX-512 ) | **1 392** | **262** | **1 322** | **281** | **6 223** | **1 180** |
| old C++ ( `-march=native`, Highway ), 8 thr | | 146 | | 211 | | 733 |

( 8-thread rows vary by ~30 % between runs today: 301 then 408 for the catalogue on uniform 2D. )

The cause:

* The lmo tree holds only `precompiled-kernels/cpu-x86-64` ( the level decision of 2026-10-20 dropped v2-v4 ), so the Xeon
  ( v4 ) behaves as x86-64: its `.so` carry no ymm/zmm instruction at all ( objdump ).
* `Engine2Reg.h` keeps the cell in `SimdVec<double,8>` and lives on `permute` / rotations. asimd has a register `permute`
  for 8 doubles only in ONE register: AVX-512. In SSE2 and in AVX2 ( two ymm ) the 8-lane permute goes through memory
  ( `SimdOps.h`: "a split into two half-registers cannot move a lane from one half to the other" ): scalar speed. Hence v3
  gains nothing and only v4 does ( 1.6x on one thread ). The old engine was `-march=native` with Highway `FixedTag<float, 8>`.
* Why the level decision dropped v4: `catalogue.level_workloads` has ONE workload of seven on the 2D register kernel, and it
  builds a new diagram each time ( the tree, insensitive to the level, weighs as much as the cells at 8 threads ). The 1.6x
  is diluted to 2 %. ( Their `.measures.value` is right: `catalogue.py` imports the submodules, whose objects are loom
  `Tensor`s; only the public `sdot` namespace hands out arrays, which is what broke the benches. )
* Not explained yet: with native kernels, 8 threads are still 1.8x the old engine on uniform 2D ( 262 against 146 ), 1.3x on
  lines, 1.6x on lines equal; and the 1-to-8-thread scaling is 5.3x. A profile of the native kernel is the next step.

### The AVX2 permute ( 2026-10-26 )

`asimd/detail/ops/X86.h`, AVX2 without AVX-512: two REGISTER variants for `double x 8` ( two ymm ) with a one-ymm index or
mask -- `permute` ( per output half: one `vpermd` to duplicate the index, two `vpermps` on the pair index, one `blendvpd` on
bit 2 ) and `select` with a 32-bit lane mask ( `vpmovsxdq` + `blendvpd` per half ). Before, both fell to the generic form
( a rank probe: rank 0 at v3 ); now rank 20. Checked against a scalar loop on 200 000 random cases at x86-64, v3 and v4
( indices outside [ 0, 8 ) included ): no mismatch. Same diagrams, ns per seed, lmo:

| level | uniform 1e6, 1 thr | 8 thr | lines Voronoi, 1 thr | 8 thr | lines equal, 1 thr | 8 thr |
|---|---|---|---|---|---|---|
| x86-64 | 2 453 | 520 | 2 252 | 601 | 8 853 | 1 898 |
| x86-64-v3, before | 2 285 | 450 | 2 178 | 483 | 7 713 | 1 422 |
| **x86-64-v3, after** | **1 798** | **342** | **1 718** | **355** | **7 530** | **1 377** |
| x86-64-v4 ( native ) | 1 385 | 257 | 1 341 | 294 | 6 226 | 1 132 |

AVX2 now gains 1.27-1.3x over SSE2 on the Voronoi cases ( it gained nothing ), about half the way to AVX-512. `lines equal`
barely moves: its cells spend their time elsewhere ( the excursions past eight vertices, the weights ). The catalogue still
ships x86-64 only, so users get none of this until the level decision keeps v3 / v4 ( `level_workloads`, see above ).

### The level decision with cells workloads ( 2026-10-26 )

`catalogue.level_workloads` gained three workloads that time the cells of ONE diagram per level ( `_cells`: built on the warm-up
call at that level, then the weights set anew and measured; no tree ): `cells 2D 1e6 Voronoi`, `cells 2D 1e6 Laguerre`,
`cells 3D 2e5`. The decision measured without building ( `~/Projects/exp_levels_lmo.py`, `_time_levels` + `choose_levels` ),
with the AVX2 permute in, on lmo while another job held a core ( load ~5 ):

| workload | v4 | v3 | v2 | x86-64 |
|---|---|---|---|---|
| cells 2D 1e6 Voronoi | 0.665 | 0.684 | 1.068 | 1 |
| cells 2D 1e6 Laguerre | 0.648 | 0.707 | 0.848 | 1 |
| cells 3D 2e5 | 0.833 | 0.794 | 0.879 | 1 |
| measures 2D 1e6 Laguerre | 0.917 | 0.912 | 1.017 | 1 |
| measures 3D 2e5 | 0.988 | 0.943 | 0.978 | 1 |
| moments 3D 2e5 | 0.998 | 0.961 | 0.988 | 1 |
| hessian 2D 1e6 | 0.969 | 0.948 | 1.012 | 1 |
| image measures 2D 1e6 | 1.049 | 1.012 | 1.007 | 1 |
| SdotPlan1d 1e6 | 0.901 | 0.889 | 0.924 | 1 |
| OT solve 2D 2e4 image | 0.925 ( spread 1.06 ) | 0.968 | 1.017 | 1 |

**Kept: x86-64-v3 and x86-64.** v4 is not: over v3 it gains only on the 2D cells ( 0.97, 0.92 ) and loses elsewhere ( the
AVX-512 clock on the Skylake-W ), under the 5 % threshold. With all threads the AVX2 kernel is now within 3-9 % of the AVX-512
one on the 2D cells ( on one thread it was 1.3x: 1 798 against 1 385 ns per seed ). The next `make wheels` decides again ( no
`levels.json` in the lmo tree's `precompiled-kernels/` ) -- on a quiet machine, so that v3 is not dropped as noise.

### Old engine against new, same day ( 2026-10-26 )

The old 2D engine rebuilt on lmo ( `~/nsdot/solvers_des_familles`, `main_diagramme.cpp` + `src/bench`, asimd of `~/nsdot/SIMD`,
`g++ -O3 -march=native`, in `/data/old-engine`, 12 s ). New: `~/Projects/exp_profile_cells_lmo.py` ( the cells of one diagram in
a loop, kernels compiled for the machine ). 2D, double, ns per seed; an otrec job held one core.

| | old, 1 thr | old, 8 thr | new v4, 1 thr | new v4, 8 thr | new v3, 1 thr | new v3, 8 thr | CGAL |
|---|---|---|---|---|---|---|---|
| uniform 1e6 | **1 029-1 063** | 184-188 | 1 336-1 394 | 257 | 1 798 | 342 | 1 499 |
| + `seed_plane` inlined | | | **1 172-1 234** | **225** | 1 737 | 343 | |
| lines Voronoi | 1 021 | 176 | 1 341 | 294 | 1 718 | 355 | 1 357 |
| lines equal | 4 074 | 683 | 6 226 | 1 132 | 7 530 | 1 377 | 1 099 |
| tree, uniform 1e6 | 322 ms | 316 ms | 1 434 ms | ~480 ms | | | |

So the memory was right: the old engine beat CGAL on one thread ( 1.05 µs against 1.5 ) and still does. The gap of the new one
is PER THREAD, not in the scaling ( 1 -> 8 threads: old 5.65x, new 5.4x ); the 146 ns of the old campaign are 184 today
( the busy core, or a different asimd ).

`perf stat`, per seed and per diagram ( uniform 1e6, one thread, the difference of two run lengths ):

| | cycles | instructions | IPC | branch misses | L1 load misses |
|---|---|---|---|---|---|
| old | 4 040 | 5 800 | 1.44 | 48 | 10 |
| new | 5 310 | 8 310 | 1.56 | 37 | 40 |
| new, `seed_plane` inlined | 4 680 | 7 690 | 1.64 | 35 | 36 |

`perf record`: the old kernel is one inlined function ( `moteur_depuis` 87 %, `excursion` 7 % ); the new one called `seed_plane`
out of line for every proposed neighbour ( 5.9 % ) and `engine_run` ( 1.1 % ). Forcing `seed_plane` inline
( `cell/Providers.h`, `__forceinline__` on the card ) gives -12 % on one thread and -12 % on 8 ( 257 -> 225 ); the new engine is then
1.13x the old on one thread, 1.22x on 8. What is left, to look at next: **+32 % instructions** ( the generic provider: 64-bit
indices `BspSeeds<double, long long, ...>`, the periodic / group / memo / slack hooks, the plane rounding ) and **3.6x the L1
misses** ( the layout of the tree and of the seeds; the new tree also takes 4.5x as long to build ). OpenBLAS's spinning threads
( 6 % of the samples ) cost nothing measurable ( 225 against 228 ns at 8 threads with `OPENBLAS_NUM_THREADS=1` ). On AVX2 the
inline barely helps ( 1 798 -> 1 737 ): there the emulated 8-lane permute dominates.

### Where the per-thread gap comes from ( 2026-10-26 )

**Same work.** Counters in both providers ( temporary, removed ), per seed, uniform 1e6, 2D:

| | nodes popped | pass the pruning | leaves opened | planes proposed |
|---|---|---|---|---|
| old | 47.39 | 27.21 | 4.01 | 29.65 |
| new | 47.39 | 27.20 | 4.01 | 29.66 |

The two traverse the same tree and propose the same planes: the gap is the COST PER OPERATION ( ~19 instructions more per
node or plane ), not the algorithm.

**Where the instructions go** ( per seed, one thread ):

| | instructions | L1 misses | ns per seed |
|---|---|---|---|
| old | 5 800 | 10 | 1 029-1 063 |
| new | 8 310 | 40 | 1 336-1 394 |
| new, `seed_plane` inlined ( kept ) | 7 690 | 36 | 1 172-1 234 |
| + contiguous reads, strides known at compile time ( experiment, reverted ) | 6 930 | 43 | 1 131-1 179 |

* `seed_plane` out of line: 620 instructions per seed, -12 % of the time. Kept ( `cell/Providers.h` ).
* The type-erased storage of `sdot_engine` ( `engine/Seeds.h`: `V1/V2/V3`, raw pointer and BYTE strides read at run time, so
  that the engine is compiled once for every layout ): 750 instructions per seed ( an `imul` per index ), but only ~4 % of the
  time ( 228 against 225 ns at 8 threads: nothing ). Reverted: assuming contiguity breaks the "any layout" contract; a
  `contiguous` template flag chosen by `PowerDiagram.py` ( which already spells `BspSeeds<...>` ) would keep it.
* What remains: +20 % instructions ( the 64-bit indices of the stack and the slices, the packed `( n << 6 ) | h` stack entries
  and the perfect-tree child arithmetic, the `slack` terms, the empty-slot test ) and **4x the L1 misses**, untouched by the
  strides. The likely cause is the layout of the tree: the old `Node` is ONE cache line ( box, majorant, `beg / end / right`,
  `static_assert` ), the new tree is split into `node_begin`, `node_end` ( int64 ), `node_box` ( and `node_wa / node_wb` in
  Laguerre ): three lines per popped node in Voronoi, five in Laguerre, for 47 pops per seed. Not tested: it needs a packed
  node record built by `AaBsp` ( a change of the tree's tensors, CPU and card ).

### A packed node record: fewer L1 misses, no time ( 2026-10-26 )

Experiment ( kept as `bench/experiments/packed-nodes.patch`, reverted ): `AaBsp.node_pack`, eight doubles per node on a 64-byte
boundary ( `lo`, `hi`, `begin`, `end` ), built by the host construction ( unbound on the card ), read by `ProviderBsp` through
`nbeg / nend / nbox` when bound ( `BspSeeds<..., PACKED>` for `sdot_engine` ). Same measures as the plain sweep in 2D / 3D,
Voronoi / Laguerre.

On a QUIET lmo ( the earlier 8-thread numbers of this page were taken while an otrec job held a core: they were 20-30 %
pessimistic ), uniform 1e6, double, ns per seed:

| | 1 thr | 8 thr | Laguerre 1 thr | Laguerre 8 thr | L1 misses / seed |
|---|---|---|---|---|---|
| old engine | **1 015** | **143** | **1 191** | **167** | 10 |
| new, split tree ( current, `seed_plane` inlined ) | 1 166 | 169 | 1 432 | 202 | 36 |
| new, packed nodes | 1 193 | 172 | 1 426 | 206 | **7** |

The packed record divides the L1 misses by five and changes nothing in time, on one thread or eight: those misses hit L2 and
the out-of-order core hides them. The remaining gap -- 1.15x on one thread, 1.18x on eight, 1.2x in Laguerre -- is the
instructions ( ~7 700 against 5 800 per seed ), not the memory. Left to try there: the 64-bit indices and stack entries, the
perfect-tree child arithmetic and empty-slot test, the `slack` terms, a `contiguous` flag for `sdot_engine`'s strides ( -10 %
instructions measured, ~-4 % time ).

### The extra instructions ( 2026-10-26 )

Instruction samples by source file ( `perf record -e instructions`, `-g` builds ): the walk itself ( `Providers.h` ) costs no
more than the old `FournisseurBsp2D.h`; what the old engine has no counterpart for is `engine/Seeds.h`, the type-erased
accessors of `sdot_engine` ( ~11 % of the per-diagram instructions ). Tried one at a time, quiet lmo, uniform 1e6, ns per seed:

| step | 1 thr | 8 thr | Laguerre 1 thr | Laguerre 8 thr | instr. / seed |
|---|---|---|---|---|---|
| start ( `seed_plane` inlined ) | 1 166 | 169 | 1 432 | 202 | 7 690 |
| + contiguous reads | 1 118 | 163 | 1 309-1 326 | 190 | |
| + `slack` at compile time ( **kept**, both ) | **1 072** | **158** | **1 289-1 302** | **185** | **6 670** |
| + 32-bit stack and slice indices ( reverted ) | 1 065 | 155 | 1 283-1 286 | 186-187 | |
| + tree depth by `clz` instead of a loop per cell ( reverted ) | 1 067-1 070 | 157-158 | 1 279-1 286 | 185 | |
| old engine | 1 015 | 143 | 1 191 | 167 | 5 800 |

Kept:

* **`BspSeeds<..., CONTIG>`** ( `engine/Seeds.h` ): `A1 / A2 / A3` read with compile-time strides when every array the walk
  reads is laid out contiguously, which `engine::dense_view` tells from the view's TYPE ( its `Strides` are the default
  `contiguous_strides` ) and `PowerDiagram.ffi_units` from the call's arguments ( `not runtime_strides and layout.is_identity` ),
  so that both name the same unit variant. A strided storage keeps the run-time strides.
* **`ProviderBsp<..., SLACK>`**: the widened boxes only for the canonical cell ( `diagram/Ops.h` asks
  `provider<TK, true>` ); the margin, a member read at run time even when zero, cost 5 % of a 2D Voronoi diagram.

Not worth it: 32-bit indices ( 1-2 %, in the noise, and a size limit ), the depth loop ( nothing ). The gap to the old engine is
now 1.06x on one thread, 1.10x on eight, 1.08-1.11x in Laguerre; what is left is spread thin ( `Plane.h` and `Local2.h`, the
bisector and the excursion past eight vertices, ~2 % each ).

### Building the CGAL programs on lmo

Now `build_baselines.py` ( `cgal/build.sh`, the old campaign's headers copied into `cgal/nsdot-src/` ); the first build, by hand:

```bash
ssh lmo 'bash -s' <<'B'
mkdir -p /data/cgal-bench && cd /data/cgal-bench
test -d cgal/.git || git clone --depth 1 https://github.com/CGAL/cgal.git cgal
# the old campaign's headers ( power_3d --cells ): rsync ~/nsdot/2d_des_familles/src/{geometry,util} to nsdot-src/
P=/data/mamba/envs/nsdot; incs=(); for d in cgal/*/include; do incs+=( -I"$d" ); done
for b in power_2d power_3d; do
  g++ -std=c++20 -O3 -march=native -DNDEBUG -DCGAL_NDEBUG "${incs[@]}" -Insdot-src -I$P/include \
      /data/Projects-rsync/sdot-ffi/bench/cgal/$b.cpp -L$P/lib -Wl,-rpath,$P/lib -lgmp -lmpfr -o $b
done
B
```

# geogram ( A3, B9 ), 2026-10-27

## What geogram has, and what is compared

geogram 6d674b93 ( 2026-10-24 ) + exploragram 9550cc6 ( 2026-04-21 ). **The transport code is not in the geogram
repository**: it is exploragram ( https://github.com/BrunoLevy/exploragram ), to clone into `src/lib/exploragram`, where
geogram's CMake picks it up.

| piece | what it computes | notes |
|---|---|---|
| `geogram/voronoi/RVD.h` `RestrictedVoronoiDiagram` | the cells of a Voronoi diagram of dimension d+1 restricted to a simplicial mesh ( surface or tetrahedra ), with callbacks per cell ∩ simplex | a power diagram is the Voronoi diagram of the lifted seeds `( p, sqrt( W - w ) )`, `W = max w`: the same cells as sdot's `|x - p|² - w` |
| `delaunay "BPOW2d"` | 2D regular triangulation ( the 3D lifted Delaunay ) | sequential |
| `delaunay "PDEL"` | 3D regular triangulation, parallel | 1.7x on 8 threads at 1e6 ( 9.8 -> 5.6 µs per seed ) |
| exploragram `OptimalTransportMap2d / 3d` | semi-discrete OT from a mesh ( constant or P1 density, vertex attribute `weight` ) to diracs | `optimize()`: L-BFGS ( HLBFGS, m = 7 ) by default; `set_Newton( true )`: KMT damped Newton ( `optimize_full_Newton` ); `optimize_levels()`: the multilevel scheme ( BRIO levels, the weights of a level fitted on the previous one by local quadratic least squares ) |
| `OptimalTransportOnSurface` | the same, the target a triangulated surface in 3D | not compared ( sdot's surfaces: `CodimMesh` ) |
| `examples/exploragram/compute_OTM` | a tet mesh to a tet mesh ( it samples the second by a CVT ), morph output | multilevel L-BFGS, `epsilon = 0.01`; it cannot load a point cloud and prints no per-part timings: hence the driver `geogram/geogram_ot.cpp` |

exploragram's Newton ( `optimal_transport.cpp::optimize_full_Newton` ):

* direction: the Hessian ( the P1 Laplacian of the cells ) assembled by OpenNL, solved by **Jacobi-preconditioned CG to
  `|Ax - b| / |b| <= 1e-3`, 1000 iterations at most** ( `OT_PRECG`, the default; `OT_SUPERLU` / `OT_CHOLMOD` are OpenNL
  extensions loaded at run time, not built here );
* step: KMT, halve from 1 until the smallest cell is `>= epsilon0` ( half the smallest of the first diagram, or of nu ) and
  `|g| <= ( 1 - alpha / 2 ) |g_k|`, 100 halvings at most ( `linesearch_init_iter` can start lower, off by default );
* stop: `|m - nu|_2 < sqrt( n ) eps nu`, **an RMS of the relative error**, `eps = 0.01` by default -- sdot stops on the max;
* an empty cell at the start of an iteration is fatal ( "There were empty cells", the solve returns );
* each iteration evaluates the Hessian on the triangulation of the accepted step ( no new triangulation ), then one
  triangulation + cells per step trial: `diagrams = 1 + trials`.

Threads: the cells are computed in parallel over a Hilbert partition of the domain mesh into `Process::max_threads()` parts
( a mesh of 2 triangles gives no parallelism: the driver meshes the box finer, see below ); PDEL is parallel, BPOW2d is
not; OpenNL's CG uses OpenMP; the rest ( lifting, line search ) is sequential.

## The driver and the bench

`geogram/geogram_ot.cpp` ( built by `geogram/build.sh` into `/data/geogram-bench` on lmo: static, `-O3 -march=native`, the
library's own `-ffp-contract=off` ) loads a binary cloud, meshes the unit box in `g^d` sub-boxes ( 2 triangles / 6 Kuhn
tetrahedra each; `g` the smallest giving 4 simplices per thread: 1 thread 2D g = 2 / 3D g = 1, 8 threads g = 4 / 2; g = 8
was not faster ), and times:

* `--level=diagram`: `delaunay->set_vertices` ( the regular triangulation ) + `funcgrad` ( the restricted cells, their
  measures ), min of 3 after a warm-up. **No tree / triangulation is reused between repetitions**: it is "from scratch",
  to put next to sdot's cells + tree.
* `--level=newton`: a copy of `optimize_full_Newton` that changes ONLY the stop, to sdot's `max |m - nu| / nu <= 1e-6`;
  wall time of the solve from zero weights, the mesh and the OTM object built before the clock ( 1-40 ms ).

`bench_geogram.py` ( same params as `bench_cgal.py`, plus `level`, `threads`, `grid`, `multilevel`, `linsolve_eps`, `rtol`,
`max_iter`, `check` ) recomputes with sdot, untimed: the measures at the same weights ( diagram ), the residual at the weights
geogram returns ( newton ).

**Same result on both sides** ( `--check=yes`, sdot `kernel_dtype = FP64` ): max `|m_geogram - m_sdot| * n` is 1e-13 ( 3D ) to
3e-10 ( 2D, the lifting `sqrt( W - w )` costs a few digits ), the same empty cells in Laguerre diagrams ( 68 252 of 1e6 at
`wscale = 0.3` ); `lines_equal` 5e-6 ( the coincident seeds of that file, which sdot separates by symbolic perturbation and
geogram by its own ). The residuals of the Newton runs below are sdot's.

## Results, lmo ( Xeon W-2145, 8 cores ), double

Caveat: another session's test runs ( `--no-queue` ) compiled on lmo during these runs ( load 1-4 ). The 1-thread diagram
numbers repeat to 3 % over three runs; the Newton times are one run each ( min of 3 for the uniform ones ) and could be a
little pessimistic.

### A3: one diagram, ns per seed ( geogram: triangulation + cells )

sdot: `lmo.json` ( cells, the tree apart: `t_bsp_build / n` in parentheses ); CGAL: the A1 CELLS table.

| case | n | geogram 1 thr | geogram 8 thr | sdot 1 thr ( + tree ) | sdot 8 thr ( + tree ) | CGAL |
|---|---|---|---|---|---|---|
| 2D uniform, Voronoi | 1e6 | 3 060 ( 2 072 + 985 ) | 1 476 ( 1 323 + 150 ) | 1 074 ( + 1 543 ) | 156 ( + 502 ) | 1 499 |
| 2D uniform, Laguerre 0.3 | 1e6 | 2 933 ( 1 975 + 942 ) | 1 549 ( 1 359 + 189 ) | | | |
| 2D lines, Voronoi | 1e5 | 2 037 ( 1 403 + 633 ) | 1 547 ( 1 254 + 269 ) | 1 048 ( + 1 693 ) | 197 ( + 1 210 ) | 1 357 |
| 2D lines, equal | 1e5 | 2 132 ( 1 433 + 698 ) | 1 452 ( 1 165 + 209 ) | 4 298 ( + 2 050 ) | 657 ( + 1 442 ) | 1 099 |
| 3D uniform, Voronoi | 1e6 | 17 177 ( 9 793 + 7 380 ) | 6 944 ( 5 581 + 1 346 ) | 9 755 ( + 1 885 ) | 1 367 ( + 512 ) | 13 660 |
| 3D planes, Voronoi | 1e5 | 16 252 ( 9 001 + 7 250 ) | 7 298 ( 5 486 + 1 686 ) | 9 515 ( + 2 026 ) | 1 411 ( + 1 404 ) | 13 335 |
| 3D planes, equal | 1e5 | 16 229 ( 8 585 + 7 644 ) | 7 291 ( 5 638 + 1 652 ) | 18 269 ( + 2 353 ) | 2 804 ( + 1 650 ) | 13 269 |

* Sequential, from scratch, geogram is a little behind sdot ( 2D 3.1 against 2.6 µs, 3D 17.2 against 11.6 ) and behind CGAL,
  except on the `equal` clouds where sdot's cells are slow ( 2D 6.3, 3D 20.6 µs ).
* On 8 threads geogram's cells scale ( 6-7x ) but its triangulation does not ( 2D sequential, 3D 1.7x ): 3D uniform 6.9
  µs against sdot 1.9 ( 3.7x ), 2D 1.5 against 0.66 ( 2.2x ). Where sdot wins comes from not triangulating.
* geogram does not reuse anything between diagrams ( no SEQUENCE scenario: each Newton trial re-triangulates ), so its
  per-diagram cost in a solve is the "from scratch" one.

### B9: Newton from zero weights, 1e5, 8 threads, stop `max |m - nu| / nu <= 1e-6`

| case | geogram KMT | iterations / diagrams | time split ( triangulations, cells, Hessian, CG ) | sdot ( `lmo.json`, `--step=auto` ) |
|---|---|---|---|---|
| 2D uniform | 7.26 s | 6 / 8 | 0.97, 0.16, 0.26, **5.85** ( 5 673 CG iterations ) | 2.45 s ( 6 / 7 ) |
| 3D uniform | 8.56 s | 5 / 8 | 4.42, 1.24, 1.12, 1.77 | 2.48 s ( 5 / 6 ) |
| 2D lines, Voronoi | **not converged**: 100 iterations, 639 diagrams, 200 s, residual 1.8e-5 | | 79.6, 17.0, 4.3, 98.9 | 3.83 s ( 12 / 19 ) |
| 3D planes, Voronoi | 27.4 s | 13 / 27 | 15.0, 4.6, 3.0, 4.8 | 6.71 s ( 9 / 17 ) |
| 2D lines, equal ( from 0 ) | **fails**: 100 iterations, 5 715 diagrams, 952 s, residual 1.6e3 | | | ( ablation: 14.2 s with aggregation ) |
| 3D planes, equal ( from 0 ) | 26.4 s | 13 / 27 | | ( ablation: 14.7 s ) |

With the CG at 1e-6 ( `--linsolve-eps=1e-6` ) nothing changes: 2D uniform 7.80 s ( 6 / 8 ), lines still not converged
( 100 iterations, 676 diagrams, 205 s, residual 2.1e-4 ). **The CG stops on its 1000-iteration cap, not on its tolerance**:
100 000 CG iterations for 100 Newton steps on the lines, 6 000 for 6 on the uniform cloud ( 5 673 at 1e-3 ). Jacobi CG on this
Laplacian at 1e5 unknowns needs more; the direction is truncated, which also explains part of the stall on the lines. A
fairer B9 row would give geogram a direct solver ( CHOLMOD through OpenNL ) or a higher cap ( `set_linsolve_maxiter` ).

* The multilevel scheme ( `optimize_levels` with `set_Newton( true )`, `--multilevel=yes` ) **fails on every case**: after
  the fit of the weights from one level to the next, the first diagram of the finer level has empty cells, which stops
  exploragram's Newton ( "There were empty cells" ) -- with L-BFGS ( `compute_OTM`'s default ) an empty cell is not fatal.
  To ask Lévy: whether his multilevel Newton ( the cosmology code ) uses another fit or a damping of the fitted weights.
* KMT on the lines needs many halvings ( 6.4 trials per iteration ) and stalls at 1e-5: sdot's limits step and log
  residual are what gets it there in 19 diagrams ( ablation, B1-B3 ). On the duplicates of `lines_equal` it cannot work
  ( no aggregation ).
* In 2D the CG ( Jacobi, 1e-3 ) is 80 % of the time; in 3D the sequential-ish triangulation is half.

## What is fair, what is not

Fair: the same clouds, the same box, the same cells ( checked to 1e-10 ), double on both sides, threads given to both, times
without I/O, the residual of both measured by sdot.

To say with the numbers:

* **stopping rule**: geogram's own is an RMS at 1 %; the driver swaps in sdot's max at 1e-6 ( one line of the loop ).
  With geogram's rule ( `eps = 1e-6` RMS ) it would stop earlier, never later; at its default 1 % it is not the same problem.
* **domain**: geogram integrates over a simplicial mesh, sdot over a box ( half-spaces ). The mesh is part of geogram's
  method ( it gives it any domain and P1 densities, sdot's `Mesh` / `Image` do the same ); its cost per simplex boundary is
  small here ( g = 8 instead of 2: +8 % in 3D at 8 threads ).
* **threads**: geogram's 2D triangulation is sequential; its parallelism is the RVD's and PDEL's. sdot is given 8 pinned
  threads, geogram 8 threads ( its own pool ) and 8 OpenMP threads for its CG.
* **linear solver**: exploragram's default ( Jacobi CG, 1e-3 ) is what is timed; CHOLMOD / SuperLU are possible through
  OpenNL extensions ( not built ). sdot uses AMG / Cholesky.
* **what geogram is for**: its recommended path is multilevel L-BFGS at 1 % ( `compute_OTM` ), not a 1e-6 Newton. A table
  "geogram's defaults, time to 1 % RMS" against sdot at the same rule would be the fair "as the author uses it" row; not run.
* Before publishing: send Lévy the driver and the numbers ( the protocol of the common rules ).

## Building

```bash
errand run --env lmo-numpy -x sdot-ffi/bench/build_baselines.py only=geogram   # what page.py does before its jobs
```

( lmo's first build was by hand, `GEOGRAM_SRC=/data/Code/geogram build.sh /data/geogram-bench`; a `SOURCES` stamp there
tells `build_baselines.py` it is current. )

`build.sh` clones exploragram into the geogram tree if missing, initialises the submodules, builds `geogram` and `exploragram`
( `Linux64-gcc` platform, Release, no graphics, no Lua ) and links the driver ( `-ltbb`: the system's libstdc++ parallel
algorithms ). `VERSION` next to the binary records both commits; `bench_geogram.py` stores it in the results.
