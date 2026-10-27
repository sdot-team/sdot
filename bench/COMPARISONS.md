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
