"""
LinearSolver — sparse linear system solver for sdot.

Backends (tried in order unless `method` is forced):
  1. 'amg'    — PyAMG algebraic multigrid + CG (ideal for Laplacian-like systems)
  2. 'direct' — scipy.sparse.linalg.spsolve

Dask arrays are computed before the solve (scipy/pyamg don't understand Dask chunks).

Usage
-----
    solver = LinearSolver()                  # auto-select backend
    x = solver.solve(rows, cols, vals, rhs, n)

    # or, reuse the same AMG hierarchy for multiple RHS (same sparsity):
    solver.setup(rows, cols, vals, n)
    x1 = solver.apply(rhs1)
    x2 = solver.apply(rhs2)
"""

import numpy

class LinearSolver:
    """
    Sparse linear solver.

    Parameters
    ----------
    method : 'auto' | 'amg' | 'direct'
        Backend selection.  'auto' tries amg first, falls back to direct.
    tol : float
        Relative tolerance for iterative solvers.
    max_iter : int | None
        Maximum iterations for iterative solvers.
    """

    def __init__( self, method='auto', tol=1e-11, max_iter = 100 ):
        self.max_iter = max_iter
        self.method   = method
        self.tol      = tol

        self._M_csr   = None   # cached sparse matrix
        self._ml      = None   # cached AMG hierarchy

    # ── public API ────────────────────────────────────────────────────────────

    def solve( self, rows, cols, vals, rhs, n ):
        """
        Solve M x = rhs where M is given in COO format.

        Parameters
        ----------
        rows, cols, vals : array-like (possibly Dask)
            COO triplets of the sparse matrix.
        rhs : array-like
            Right-hand side vector, length n.
        n : int
            Matrix size.

        Returns
        -------
        x : numpy array, length n.
        """
        self.setup( rows, cols, vals, n )
        return self.apply( rhs )

    def setup( self, rows, cols, vals, n ):
        """
        Build and cache the sparse matrix (and AMG hierarchy if applicable).
        Call this once when the sparsity pattern / values change.
        """
        self._M_csr = _build_csr( rows, cols, vals, n )
        self._ml = None # invalidate cached hierarchy

        if self._use_amg():
            try:
                import pyamg
                self._ml = pyamg.smoothed_aggregation_solver(
                    self._M_csr,
                    coarse_solver='pinv',
                )
            except Exception:
                self._ml = None   # fall through to direct

    def apply(self, rhs):
        """Solve with the previously set-up matrix."""
        if self._M_csr is None:
            raise RuntimeError("Call setup() before apply()")

        rhs_np = _to_numpy(rhs)

        if self._ml is not None:
            x, info = self._ml.solve(
                rhs_np,
                tol      = self.tol,
                maxiter  = self.max_iter,
                return_info = True,
            )
            if info == 0:
                return x
            # AMG didn't converge — fall through to direct

        return self._direct(rhs_np)

    # ── internals ─────────────────────────────────────────────────────────────

    def _use_amg(self):
        if self.method == 'amg':
            return True
        if self.method == 'direct':
            return False
        # 'auto': use AMG if PyAMG is importable
        try:
            import pyamg  # noqa: F401
            return True
        except ImportError:
            return False

    def _direct(self, rhs_np):
        from scipy.sparse.linalg import spsolve
        return spsolve(self._M_csr, rhs_np)


def _to_numpy(a):
    """Materialise a Dask / JAX / Torch array into a plain numpy array."""
    if hasattr(a, 'compute'):        # dask
        return numpy.asarray(a.compute())
    if hasattr(a, 'numpy'):          # torch
        return a.numpy()
    try:
        import jax.numpy as jnp
        if isinstance(a, jnp.ndarray):
            return numpy.asarray(a)
    except ImportError:
        pass
    return numpy.asarray(a)


def _build_csr(rows, cols, vals, n):
    from scipy.sparse import coo_matrix
    rows = _to_numpy(rows).astype(numpy.int32)
    cols = _to_numpy(cols).astype(numpy.int32)
    vals = _to_numpy(vals)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

