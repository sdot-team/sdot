# Plan-based starts for regular diracs → contrasted image (prototype, 2026-11-20)

`exp_plan_init.py` (not committed). The source is N×N grid diracs with equal masses. The target is an image of 1024² pixels with a floor of 1e-3: bands at 30°, a contrasted sine `(½+½ sin sin)^4`, and a white star on black. Full tables are in `results_tables.md`; figures are `cells_*.png` (starting cells before repair, cells under 10 % of their mass in blue).

The general Newton (`SumOfDiracs`) runs with `continuation = "never"` and `Tuning( keep_start = True )`, so the start is the only help. The coarse grid is N/2, solved by sdot's defaults; its time and iterations come on top of the fine solve. A **front repair** comes before Newton: empty cells (m·n < 1e-4) that have a non-empty neighbour get their weight raised, with a doubling step. Without it, every start that has an empty cell fails at iteration 1. sdot's own warm repair does not fix these starts, and without `keep_start` sdot swaps them for the Voronoi start.

## Newton iterations on the fine grid ("–" = not converged in 200, or start unusable)

| image | N | sdot default (its continuation) | w = 0 | cubic interp. | **local plan** | global Knothe |
|---|---|---|---|---|---|---|
| sine  | 32  | 12 | 11 | 7  | **5**  | – |
| sine  | 64  | 12 | 15 | 7  | **5**  | – |
| sine  | 128 | 16 | 21 | 8  | **5**  | – |
| sine  | 256 | 18 | 42 | 10 | **6**  | – |
| bands | 32  | 16 | 24 | 33 | **10** | 39 |
| bands | 64  | 21 | 87 | 37 | **25** | – |
| bands | 128 | 28 | –  | –  | 72     | – |
| bands | 256 | 40 | –  | –  | –      | – |
| star  | 32  | 32 | 143 | 9 | **7**  | – |
| star  | 64  | 43 | –  | –  | **15** | – |
| star  | 128 | 54 | –  | 20 | –      | – |
| star  | 256 | 69 | –  | –  | –      | – |

The coarse solve also costs iterations (12–16 for the sine, 21–54 for the sharp images), but each of them is on 4× fewer cells.

## What the figures and residuals say

- **Smooth contrast (sine):** the local plan wins at every N, with 5–6 iterations (2–3 damped) against 16–18 for sdot's default. The least-squares face residual stays near 0.3 h whatever N is.
- **Sharp edges (bands, star):** the local plan works up to N = 64, then degrades. Its face residual grows with N (bands 0.30 → 0.41 → 0.57 → 0.79 h, star 0.11 → 0.14 → 0.24 → 0.60 h). The empty cells grow with it (bands 3 → 119 → 704 → 4021) and gather into clusters: thin strip cells crossing the black regions, and cells along the edges and tips. Once clusters are wide, the front repair stalls.
  - *Why:* when an edge crosses a coarse cell, the Knothe y-cuts of the left and right halves differ a lot. A 2×2 power diagram cannot represent that T-junction: its 4 face equations are consistent only if the two y-cuts are equal. Least squares averages them, which shifts faces by a fraction of h. That is fatal for white cells of width ≈ h and for the thin strips in black.
- **Global Knothe:** face residual ≈ 2 h × (N/32), an O(1) absolute error as predicted. Hundreds to tens of thousands of empty cells; unusable.
- **Cubic interpolation of w − |g|²:** good on the sine; bad at the kinks (empty clusters along the bands, failures at 64+).

## Leads

1. Recurse instead of jumping from N/2 once. Each level then corrects at its own scale, and errors do not accumulate in one step. That is what sdot's default does with ψ.
2. Choose the split so it stays representable as a power diagram: make the two y-cuts equal (a cross split), or solve a 4-dirac power-diagram problem inside each coarse cell instead of using Knothe.
3. A repair that solves the empty clusters (local Newton on the cluster plus its ring) instead of raising weights.

Side finding: `ot_solve( SumOfDiracsOnAGrid( ( N, N ) ), img )` returns status `failure` with 0 iterations on the bands and star images (floor 1e-3) for every N tested, 32 to 256. It converges on the sine.
