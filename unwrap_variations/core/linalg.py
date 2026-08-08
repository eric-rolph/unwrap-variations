"""Sparse symmetric linear algebra, written against numpy and nothing else.

The flattener in :mod:`unwrap_variations.core.flatten` has to solve ``L x = b``
with ``L`` a sparse symmetric positive semi-definite Laplacian, at up to a few
hundred thousand unknowns, inside Blender's interpreter.  Blender 5.1.2 ships
**Python 3.13.9 with numpy 2.3.4 and nothing else** -- there is no ``scipy``,
it cannot be made an optional dependency and it must not be vendored (see
``docs/ORGANIC.md`` section 0.5).  So the solver lives here, in about three
hundred lines of numpy.

Nothing in this module imports :mod:`bpy`; it is unit-tested in plain CPython.

Design notes
------------

*Why an operator, not a matrix?*  Nothing downstream needs a factorisation or a
row slice -- conjugate gradients only ever asks for ``A @ x``.  Storing the COO
triplets and applying them with :func:`numpy.bincount` therefore costs one
gather and one scatter per product, needs no index structure beyond the
triplets themselves, and skips the CSR conversion entirely.

*Why* :func:`numpy.bincount` *rather than* :func:`numpy.add.at`?  They compute
the same scatter-add, but ``bincount`` runs a tight C loop over a contiguous
weight array while ``add.at`` goes through the generic ufunc-at machinery.
Measured on the 100k-vertex cotan Laplacian this module is budgeted against,
``bincount`` is roughly an order of magnitude quicker, which is the difference
between hitting the 10 ms matvec budget and missing it by a factor of five.

*Why coalesce the triplets at construction?*  A Laplacian assembled the natural
way -- walk the triangles, emit nine entries each -- produces about 2.5
duplicates per stored entry.  Summing them once up front makes every subsequent
matvec that much cheaper, and CG performs hundreds of them.  Coalescing also
sorts the triplets by row, which turns ``bincount``'s scatter into a nearly
sequential write.

*Why never raise?*  :func:`pcg` sits underneath a Blender operator.  A singular
or indefinite system is a perfectly ordinary thing for a degenerate chart to
produce, and the honest response is to hand back the best iterate found along
with the residual it achieved so the caller can decide -- not to throw, and
emphatically not to return a silent NaN, which would become a NaN UV layer and
a corrupted ``.blend``.  Shape-contract violations are a different matter: they
are programming errors, not data, and those do raise :class:`ValueError`, in
line with :meth:`unwrap_variations.core.meshdata.UVMeshSample.with_uvs`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["SparseSym", "pcg"]


#: Largest ``n`` for which the ``row * n + col`` coalescing key stays inside a
#: signed 64-bit integer.  Comfortably above any mesh Blender can hold.
_MAX_N = 3_000_000_000

#: Diagonal entries at or below this fraction of the largest diagonal entry are
#: treated as "no useful Jacobi scaling here" and preconditioned with 1.0.  A
#: literal zero row -- an isolated vertex, a fully constrained one -- is the
#: common case, and dividing by it would poison every subsequent iterate.
_DIAG_FLOOR = 1e-300


class SparseSym:
    """A symmetric sparse matrix held as COO triplets and applied as an operator.

    The triplets are the **explicit entries of the full matrix**: an off-diagonal
    ``A[i, j]`` needs both ``(i, j, v)`` and ``(j, i, v)``, exactly as a
    triangle-by-triangle Laplacian assembly emits them.  Duplicate coordinates
    are summed at construction.  Symmetry is the caller's contract -- it is not
    imposed, because imposing it would hide an assembly bug -- and
    :meth:`symmetry_error` exists so tests can assert it cheaply.

    Entries that are out of range or non-finite are **dropped** rather than
    stored, and counted in :attr:`dropped`.  A single NaN triplet would
    otherwise turn every matvec into a vector of NaNs, and the caller has no way
    to see where it came from; dropping it degrades the operator locally and
    leaves a number to report.

    Attributes:
        n: The matrix dimension.
        nnz: Stored (coalesced) entry count.
        dropped: How many input triplets were discarded as out-of-range or
            non-finite.
    """

    __slots__ = ("_rows", "_cols", "_vals", "_n", "_dropped", "_diag")

    def __init__(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        vals: np.ndarray,
        n: int,
    ) -> None:
        """Build the operator from COO triplets.

        Args:
            rows: ``(nnz,)`` row indices.
            cols: ``(nnz,)`` column indices.
            vals: ``(nnz,)`` values.  Duplicated ``(row, col)`` pairs are summed.
            n: Matrix dimension.  Indices outside ``[0, n)`` are dropped.

        Raises:
            ValueError: If ``n`` is negative or larger than can be keyed in
                int64, or if the three triplet arrays have different lengths.
        """
        n = int(n)
        if n < 0:
            raise ValueError("matrix dimension must be non-negative, got %d" % n)
        if n > _MAX_N:
            raise ValueError("matrix dimension %d exceeds the int64 coalescing key" % n)
        r = np.asarray(rows, dtype=np.int64).reshape(-1)
        c = np.asarray(cols, dtype=np.int64).reshape(-1)
        v = np.asarray(vals, dtype=np.float64).reshape(-1)
        if not (r.shape[0] == c.shape[0] == v.shape[0]):
            raise ValueError(
                "rows/cols/vals must be the same length, got %d/%d/%d"
                % (r.shape[0], c.shape[0], v.shape[0])
            )

        self._n = n
        self._diag: np.ndarray | None = None
        if n == 0 or r.size == 0:
            self._rows = np.zeros(0, dtype=np.intp)
            self._cols = np.zeros(0, dtype=np.intp)
            self._vals = np.zeros(0, dtype=np.float64)
            self._dropped = int(r.size)
            return

        keep = (r >= 0) & (r < n) & (c >= 0) & (c < n) & np.isfinite(v)
        dropped = int(r.size - np.count_nonzero(keep))
        if dropped:
            r = r[keep]
            c = c[keep]
            v = v[keep]
        self._dropped = dropped
        if r.size == 0:
            self._rows = np.zeros(0, dtype=np.intp)
            self._cols = np.zeros(0, dtype=np.intp)
            self._vals = np.zeros(0, dtype=np.float64)
            return

        # Coalesce.  ``np.unique`` returns the keys sorted, so the stored
        # triplets come out ordered by row and then by column -- the scatter in
        # ``matvec`` is then almost sequential.
        key = r * n + c
        uniq, inverse = np.unique(key, return_inverse=True)
        summed = np.bincount(inverse.reshape(-1), weights=v, minlength=uniq.size)
        self._rows = (uniq // n).astype(np.intp)
        self._cols = (uniq - self._rows.astype(np.int64) * n).astype(np.intp)
        self._vals = summed[: uniq.size]

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_coalesced(
        cls,
        rows: np.ndarray,
        cols: np.ndarray,
        vals: np.ndarray,
        n: int,
    ) -> "SparseSym":
        """Build the operator from triplets that are already unique and in range.

        The fast path for a caller that reassembles the *same sparsity pattern*
        many times with different values -- which is exactly what the
        flattener's distortion-reweighted preconditioner does once per refresh.
        It skips the sort in :meth:`__init__`; nothing is validated beyond the
        shapes, so a caller that lies about uniqueness gets a wrong answer
        rather than an exception.

        Args:
            rows: ``(nnz,)`` row indices, all inside ``[0, n)``.
            cols: ``(nnz,)`` column indices, all inside ``[0, n)``.
            vals: ``(nnz,)`` finite values, one per unique coordinate pair.
            n: Matrix dimension.

        Returns:
            The operator.
        """
        obj = cls.__new__(cls)
        obj._n = int(n)
        obj._diag = None
        obj._dropped = 0
        obj._rows = np.ascontiguousarray(rows, dtype=np.intp)
        obj._cols = np.ascontiguousarray(cols, dtype=np.intp)
        obj._vals = np.ascontiguousarray(vals, dtype=np.float64)
        if not (obj._rows.shape[0] == obj._cols.shape[0] == obj._vals.shape[0]):
            raise ValueError("rows/cols/vals must be the same length")
        return obj

    def with_values(self, vals: np.ndarray) -> "SparseSym":
        """Return a new operator with this one's pattern and different values.

        Args:
            vals: ``(nnz,)`` replacement values, aligned with :attr:`nnz`.

        Returns:
            A new :class:`SparseSym` sharing the index arrays.

        Raises:
            ValueError: If ``vals`` is not ``(nnz,)``.
        """
        arr = np.asarray(vals, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self._vals.shape[0]:
            raise ValueError(
                "expected %d values, got %d" % (self._vals.shape[0], arr.shape[0])
            )
        return SparseSym.from_coalesced(self._rows, self._cols, arr, self._n)

    # -- sizes ----------------------------------------------------------------

    @property
    def n(self) -> int:
        """Matrix dimension."""
        return self._n

    @property
    def nnz(self) -> int:
        """Number of stored (coalesced) entries."""
        return int(self._vals.shape[0])

    @property
    def dropped(self) -> int:
        """How many input triplets were discarded as out-of-range or non-finite."""
        return self._dropped

    @property
    def rows(self) -> np.ndarray:
        """``(nnz,)`` stored row indices, sorted by ``(row, col)``."""
        return self._rows

    @property
    def cols(self) -> np.ndarray:
        """``(nnz,)`` stored column indices, sorted by ``(row, col)``."""
        return self._cols

    @property
    def values(self) -> np.ndarray:
        """``(nnz,)`` stored values, aligned with :attr:`rows` and :attr:`cols`."""
        return self._vals

    # -- the operator ---------------------------------------------------------

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Return ``A @ x`` for one or several right-hand sides.

        Args:
            x: ``(n,)`` or ``(n, k)``.  ``k`` columns are applied in one pass
                each; the flattener solves ``u`` and ``v`` together, so ``k`` is
                normally 2.

        Returns:
            The product, with the same shape as ``x``.

        Raises:
            ValueError: If ``x``'s leading dimension is not ``n`` or it has more
                than two dimensions.  A shape mismatch here is a programming
                error, not data, so it is reported loudly.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim > 2:
            raise ValueError("x must be 1- or 2-dimensional, got %dD" % arr.ndim)
        if arr.shape[0] != self._n:
            raise ValueError(
                "x has leading dimension %d, expected %d" % (arr.shape[0], self._n)
            )
        one_d = arr.ndim == 1
        block = arr.reshape(self._n, 1) if one_d else arr
        out = np.zeros((self._n, block.shape[1]), dtype=np.float64)
        if self._vals.size:
            gathered = np.empty(self._vals.shape[0], dtype=np.float64)
            for col in range(block.shape[1]):
                np.multiply(self._vals, block[self._cols, col], out=gathered)
                out[:, col] = np.bincount(self._rows, weights=gathered, minlength=self._n)[
                    : self._n
                ]
        return out.reshape(-1) if one_d else out

    def diagonal(self) -> np.ndarray:
        """Return the ``(n,)`` main diagonal.

        Computed once and cached; the operator is immutable.
        """
        if self._diag is None:
            diag = np.zeros(self._n, dtype=np.float64)
            if self._vals.size:
                on = self._rows == self._cols
                if on.any():
                    idx = self._rows[on]
                    diag = np.bincount(idx, weights=self._vals[on], minlength=self._n)[
                        : self._n
                    ]
            self._diag = diag
        return self._diag

    # -- diagnostics ----------------------------------------------------------

    def symmetry_error(self) -> float:
        """Return ``max |A[i, j] - A[j, i]|`` over the stored pattern.

        Zero for a correctly assembled symmetric operator.  Used by tests; the
        solver itself does not check, because a caller that assembles an
        unsymmetric matrix has a bug that a silent correction would hide.
        """
        if self._vals.size == 0:
            return 0.0
        key = self._rows.astype(np.int64) * self._n + self._cols.astype(np.int64)
        mirror = self._cols.astype(np.int64) * self._n + self._rows.astype(np.int64)
        where = np.searchsorted(key, mirror)
        where = np.clip(where, 0, key.shape[0] - 1)
        found = key[where] == mirror
        other = np.where(found, self._vals[where], 0.0)
        return float(np.abs(self._vals - other).max())

    def to_dense(self) -> np.ndarray:
        """Return the ``(n, n)`` dense matrix.  For tests on small systems only."""
        dense = np.zeros((self._n, self._n), dtype=np.float64)
        if self._vals.size:
            dense[self._rows, self._cols] = self._vals
        return dense

    def __repr__(self) -> str:
        """Return a compact description, for logs."""
        return "SparseSym(n=%d, nnz=%d, dropped=%d)" % (self._n, self.nnz, self._dropped)


def pcg(
    A: SparseSym,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int | None = None,
    precond: str = "jacobi",
) -> tuple[np.ndarray, int, float]:
    """Solve ``A x = b`` by Jacobi-preconditioned conjugate gradients.

    Several right-hand sides are carried through together: ``b`` may be ``(n,)``
    or ``(n, k)``, and each column runs its own scalars.  That is not just
    convenience -- the flattener solves the same Laplacian for ``u`` and ``v``,
    and sharing the matvec halves the dominant cost.

    **This function never raises on a numerically hopeless system.**  Three
    failure modes are handled explicitly, per column:

    * ``p^T A p <= 0`` -- the matrix is indefinite along the current search
      direction, so CG's step length is meaningless.  The column is frozen.
    * a non-finite iterate or residual -- the column is frozen.
    * a right-hand side that is itself non-finite -- the column never starts,
      and its reported residual is ``inf``.

    A frozen column keeps the best iterate seen so far, which is tracked
    throughout rather than assumed to be the last one: CG's residual norm is not
    monotone, and handing back a *later, worse* iterate because it happened to
    be the one the loop stopped on is a real and easy mistake.

    **Empty rows are solved around, not solved through.**  A row of ``A`` whose
    entries are all zero states ``0 = b_i``: it constrains nothing, and by
    symmetry the matching column constrains nothing either, so ``x_i`` keeps its
    initial value.  Left in the iteration such a row injects a permanent,
    irreducible residual that drags the search direction sideways -- on a
    ``diag(1, 0, 0)`` system with ``b = (1, 1, 1)`` the very first CG step makes
    the residual *worse* and the solver hands back zeros, when ``x_0 = 1`` was
    right there.  Those rows are therefore excluded from the search and their
    unsatisfiable part is added back into the reported residual, so the caller
    both gets the reachable answer and can see that the system was inconsistent.

    Args:
        A: The (assumed symmetric) operator.
        b: ``(n,)`` or ``(n, k)`` right-hand side.
        x0: Optional initial guess with ``b``'s shape.  Non-finite entries are
            replaced by zero.  Defaults to zeros.
        tol: Convergence threshold on the **relative** residual
            ``||b - A x|| / ||b||`` (absolute when ``||b|| == 0``).
        maxiter: Iteration budget.  Defaults to ``min(max(2n, 64), 4000)`` --
            enough for a well-conditioned Laplacian of any size the add-on will
            see, and bounded so a singular system cannot spin.
        precond: ``"jacobi"`` (default) or ``"none"``.  Any other value is
            treated as ``"jacobi"``; the argument exists so a caller can turn
            preconditioning off for a diagnostic comparison, not as a hook for
            arbitrary strings.

    Returns:
        ``(x, iters, residual)`` where ``x`` has ``b``'s shape and is always
        finite, ``iters`` is how many iterations ran, and ``residual`` is the
        largest relative residual over the columns -- so a single number below
        ``tol`` means every right-hand side converged.
    """
    n = int(A.n)
    b_arr = np.asarray(b, dtype=np.float64)
    if b_arr.ndim > 2:
        raise ValueError("b must be 1- or 2-dimensional, got %dD" % b_arr.ndim)
    if b_arr.shape[0] != n:
        raise ValueError("b has leading dimension %d, expected %d" % (b_arr.shape[0], n))
    one_d = b_arr.ndim == 1
    rhs = b_arr.reshape(n, 1) if one_d else b_arr
    k = rhs.shape[1]

    if n == 0 or k == 0:
        empty = np.zeros_like(rhs)
        return (empty.reshape(-1) if one_d else empty), 0, 0.0

    x = _finite_start(x0, rhs, one_d)
    poisoned = ~np.isfinite(rhs).all(axis=0)
    work = np.where(poisoned[None, :], 0.0, rhs)

    scale = np.sqrt(np.einsum("ij,ij->j", work, work))
    scale = np.where(scale > 0.0, scale, 1.0)

    inv_diag = _jacobi_inverse(A, precond)
    inert = _inert_rows(A)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return _pcg_loop(A, work, x, inv_diag, scale, poisoned, inert, tol, maxiter, one_d)


def _inert_rows(A: SparseSym) -> np.ndarray:
    """Return the ``(n,)`` mask of rows whose stored entries are all zero.

    Tested on the summed magnitudes, not on the stored-entry count: an assembly
    that emits an explicit zero -- ``(1, 1, 0.0)`` -- produces a row that is
    structurally present and numerically inert, and the two must be treated
    alike.
    """
    weight = np.bincount(A.rows, weights=np.abs(A.values), minlength=A.n) if A.nnz else None
    if weight is None:
        return np.ones(A.n, dtype=bool)
    return ~(weight[: A.n] > 0.0)


def _finite_start(x0: np.ndarray | None, rhs: np.ndarray, one_d: bool) -> np.ndarray:
    """Return a finite ``(n, k)`` starting iterate, defaulting to zeros."""
    if x0 is None:
        return np.zeros_like(rhs)
    arr = np.asarray(x0, dtype=np.float64)
    arr = arr.reshape(rhs.shape[0], 1) if one_d and arr.ndim == 1 else arr
    if arr.shape != rhs.shape:
        raise ValueError("x0 has shape %s, expected %s" % (arr.shape, rhs.shape))
    out = np.array(arr, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = 0.0
    return out


def _jacobi_inverse(A: SparseSym, precond: str) -> np.ndarray | None:
    """Return the ``(n,)`` reciprocal diagonal, or ``None`` for no preconditioner.

    Entries that are not usably positive -- a zero row, a negative diagonal from
    an indefinite assembly, a NaN -- fall back to ``1.0``.  Jacobi is only a
    scaling; a wrong scaling slows convergence, whereas dividing by zero ends it.
    """
    if precond is None or str(precond).lower() == "none":
        return None
    diag = A.diagonal()
    usable = np.isfinite(diag) & (diag > _DIAG_FLOOR)
    inv = np.ones_like(diag)
    np.divide(1.0, diag, out=inv, where=usable)
    inv[~usable] = 1.0
    return inv


def _apply_precond(inv_diag: np.ndarray | None, r: np.ndarray) -> np.ndarray:
    """Apply the (diagonal) preconditioner to every column of ``r``."""
    if inv_diag is None:
        return r.copy()
    return r * inv_diag[:, None]


def _pcg_loop(
    A: SparseSym,
    rhs: np.ndarray,
    x: np.ndarray,
    inv_diag: np.ndarray | None,
    scale: np.ndarray,
    poisoned: np.ndarray,
    inert: np.ndarray,
    tol: float,
    maxiter: int | None,
    one_d: bool,
) -> tuple[np.ndarray, int, float]:
    """Run the preconditioned CG iteration.  See :func:`pcg` for the contract."""
    n = rhs.shape[0]
    budget = min(max(2 * n, 64), 4000) if maxiter is None else max(0, int(maxiter))
    tol = max(float(tol), 0.0)

    r = rhs - A.matvec(x)
    r[:, poisoned] = 0.0
    # An inert row's residual is a constant: neither ``x_i`` nor ``(A x)_i`` can
    # ever change.  Bank it and take it out of the search.
    if inert.any():
        missed = np.einsum("ij,ij->j", r[inert], r[inert])
        r[inert] = 0.0
    else:
        missed = np.zeros(rhs.shape[1], dtype=np.float64)

    best_x = np.array(x, dtype=np.float64, copy=True)
    best_res = _column_residual(r, scale)
    best_res[poisoned] = math.inf

    def report(res: np.ndarray) -> float:
        """Fold the banked inert-row residual back into the reported number."""
        total = np.sqrt(np.maximum(res * res * scale * scale, 0.0) + missed) / scale
        return float(np.where(np.isfinite(total), total, math.inf).max())

    active = ~poisoned & (best_res > tol)
    if not active.any():
        return _finish(best_x, one_d), 0, report(best_res)

    z = _apply_precond(inv_diag, r)
    p = z.copy()
    rz = np.einsum("ij,ij->j", r, z)
    iters = 0

    for step in range(1, budget + 1):
        iters = step
        ap = A.matvec(p)
        p_ap = np.einsum("ij,ij->j", p, ap)

        # A column whose curvature has gone non-positive, or whose scalars have
        # gone non-finite, cannot take a meaningful CG step.  Freeze it and let
        # the others carry on: one degenerate chart column must not stop the
        # other from converging.
        step_ok = active & np.isfinite(p_ap) & (p_ap > 0.0) & np.isfinite(rz)
        if not step_ok.any():
            break
        alpha = np.where(step_ok, rz / np.where(p_ap != 0.0, p_ap, 1.0), 0.0)
        alpha = np.where(np.isfinite(alpha), alpha, 0.0)

        x = x + alpha[None, :] * p
        r = r - alpha[None, :] * ap

        finite_cols = np.isfinite(x).all(axis=0) & np.isfinite(r).all(axis=0)
        active &= step_ok & finite_cols

        res = _column_residual(r, scale)
        # A poisoned column is excluded from best-tracking, not merely from the
        # iteration.  Its residual reads 0 -- its right-hand side was zeroed to
        # keep the NaN out of the arithmetic -- so without this mask it would
        # "improve" on its own ``inf`` marker at the first iteration and be
        # reported as converged, which is the opposite of the truth.
        improved = finite_cols & ~poisoned & (res < best_res)
        if improved.any():
            idx = np.flatnonzero(improved)
            best_x[:, idx] = x[:, idx]
            best_res[idx] = res[idx]

        active &= best_res > tol
        if not active.any():
            break

        z = _apply_precond(inv_diag, r)
        rz_new = np.einsum("ij,ij->j", r, z)
        beta = np.where(
            active & np.isfinite(rz_new) & np.isfinite(rz) & (rz != 0.0),
            rz_new / np.where(rz != 0.0, rz, 1.0),
            0.0,
        )
        beta = np.where(np.isfinite(beta), beta, 0.0)
        p = np.where(active[None, :], z + beta[None, :] * p, 0.0)
        rz = np.where(active, rz_new, 0.0)

    return _finish(best_x, one_d), iters, report(best_res)


def _column_residual(r: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return the per-column relative residual norm, ``inf`` where non-finite."""
    sq = np.einsum("ij,ij->j", r, r)
    out = np.sqrt(np.where(np.isfinite(sq) & (sq >= 0.0), sq, np.inf)) / scale
    return np.where(np.isfinite(out), out, math.inf)


def _finish(best_x: np.ndarray, one_d: bool) -> np.ndarray:
    """Return the answer in the caller's shape, guaranteed finite."""
    bad = ~np.isfinite(best_x)
    if bad.any():
        best_x = np.where(bad, 0.0, best_x)
    return best_x.reshape(-1) if one_d else best_x
