# Ground metrics : decisions

- **Periodicity: where**: an `OtProblem` kwarg `periodicity` (so also `plan`, `ot_solve`, `w2`), stored on the problem, plus the same kwarg on `PowerDiagram`. A future space object would take `periodicity` as a shortcut.
- **Description of the periodicity**: by the group, not by a domain (symmetric in f and g). Generators are vectors or `( M, V )` with `M` orthogonal; closure and the choice of images are up to the algorithm. Shortcuts: `True`, `[ True, False ]`, `g.domain` (opposite facets → translations). Check: g's support is contained in a fundamental domain of the group. Facet pairings and slicing cuts were considered and dropped.
- **Results frame**: barycenters and gradients are unfolded in the frame of seed i (they may leave the fundamental domain). Input positions may lie outside the domain. A helper `sdot.wrap( positions, periodicity )` is provided.
- **Naming**: `w2` stays and accepts `periodicity`. `sdot.cost( f, g, ... )` comes along with the φ axis.
- **Fold / unfold**: seeds are folded into D on input (T_i is stored). Barycenters are unfolded with T_i⁻¹ and gradients with M_iᵀ. Weights are invariant.
- **Classification**: (1) space ℝᵈ or ℝᵈ/G, (2) cost φ, global or per dirac (radial functions, capture radius), (3) mass balance: unbalanced means a global sink, i.e. partial transport.
- **Partial: direction**: mass(f) < mass(g) first (Laguerre ∩ balls √w). mass(f) > mass(g) later (box constraints on the weights); explicit error until then.
- **Penalized partial (λ)**: planned in the API (sink weight = -λ), implemented after the mass-constrained case.
- **Partial: trigger**: `target_mass` on every distribution, `SumOfDiracs` included (1 by default, `None` = raw mass). No kwarg on `plan`: the final masses decide. Identical numeric `target_mass` values count as equal exactly; otherwise the masses are compared with a relative `tol`.
- **Cell construction**: one cell per real seed. The start cell is the group's Dirichlet cell centred on the seed (for a box: D recentred). It is provably sufficient whatever the weights, since a seed and its images have the same weight. Cuts come from the BSP of the real seeds, queried by translation, with no materialized seeds. Integration: the cell is cut by the facets of D and each piece is folded back into D. Moments are kept in the seed's frame. Hessian: the j = i case (a facet with its own image) is new and tested. 1D + periodicity: explicit error.
- **Supported targets**: Image and Box/Polytope first. SumOfGaussians and Mesh/spread: explicit error, then roadmap.
- **Validation**: brute-force reference: seeds copied 3ᵈ times (5ᵈ with large weights) in a large domain without periodicity, masses folded back. Compare masses, barycenters, cost, hessian, and the j = i case.
- **Order**: (1) periodicity, with `ground_metrics.md` rewritten along the classification; (2) partial with mass(f) < mass(g); (3) φ axis / radial functions; then the reverse partial and the λ variant.
