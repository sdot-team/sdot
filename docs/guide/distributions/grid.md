# Diracs on a Grid

Diracs at the centres of the cells of a regular grid, for instance one per pixel of an image:

```python
source = sdot.SumOfDiracsOnAGrid( shape, weights = None, frame = None, cell_frame = None, origin = None )
```

- `shape`: the number of cells along each axis, `( N, N )` in 2D;
- `weights`: the masses of the diracs, one per cell ( flat in C order, or of shape `shape` ), equal by default;
- `frame`: the edges of the whole grid ( rows, or their lengths for an axis-aligned grid ), `cell_frame` those of one cell.
  Neither: the longest side of the grid is 1, at `origin` ( 0 by default ). This is the convention of `Image`:
  `SumOfDiracsOnAGrid( shape )` puts a dirac at the centre of each pixel of an `Image` of that shape.

No positions are stored, only the grid. It does everything a `SumOfDiracs` does except move its diracs, and anything that
needs the positions ( `source.positions` ) gets them computed on demand.

```python
source = sdot.SumOfDiracsOnAGrid( ( 512, 512 ), weights = masses )     # masses: [ 512, 512 ]
plan = sdot.ot_solve( source, target )
```

## Why a dedicated path

On a regular grid the power cells are known in advance up to the weights. Each cell has the same candidate neighbours: the
ring of the 8 adjacent seeds, then the next rings. The solver does not need a tree, a search for neighbours or a cut cell by
cell. It only needs a few fixed operations per cell, without branches, which suit SIMD units.

The solve has three parts:

* **the cells**: each cell is computed from the ring of its 8 neighbours in one fixed sequence of operations. A local
  certificate then checks it: each vertex of a cell must also be a vertex of the cells of its neighbours. A cell that fails
  is computed again with the second ring, then, if that still fails, exactly, like an ordinary sdot cell. On smooth masses
  almost every cell passes the first ring.
* **the Newton system**: the Hessian is stored as 8 weights per cell, toward its ring of neighbours, in arrays of the
  size of the grid. The conjugate gradient is preconditioned by blurs of the residual at every scale and along both axes,
  which keep the number of iterations roughly constant in `N`.
* **coarse to fine**: the solve starts on a 64 x 64 grid, then each level starts from the interpolated solution of the
  coarser one.

The dedicated solver applies in 2D, to `N x N` grids with `N` a power of two and square cells, when the target lies on the
box of the grid. The target can be uniform ( `Box` ), an `Image`, a `SumOfGaussians`, or any density sdot integrates on a
cell. `Tuning( grid_solver = ... )` chooses the path:

| `grid_solver` | |
|---|---|
| `"auto"` ( default ) | the dedicated solver when it applies, otherwise the generic Newton on the grid's diagram |
| `"dedicated"` | the dedicated solver, or an error that says why it does not apply |
| `"cells"` | the generic Newton on the diagram of the grid ( `PowerDiagram_Grid`: an implicit tree over the grid, no positions ) |

All three give the same plan, to the tolerance of the solve.

## Performance

The timings of the three paths are on [Performance: Diracs on a Grid](../performance-grid).
