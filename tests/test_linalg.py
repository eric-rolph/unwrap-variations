"""Tests for ``unwrap_variations.core.linalg`` -- the numpy-only sparse solver.

This module exists because Blender ships numpy and nothing else
(``docs/ORGANIC.md`` section 0.5), so everything scipy would have done has to be
correct here instead.  Three properties carry the weight:

1. **It agrees with a dense solve.**  Random SPD systems and a Laplacian with a
   hand-computable answer are checked against :func:`numpy.linalg.solve`.  If
   the two disagree the operator, not the test, is wrong.
2. **It never raises and never returns a NaN.**  Singular, indefinite,
   zero-row, NaN right-hand side, empty: all of them have to come back with a
   finite iterate and an honest residual, because the caller writes the result
   into a UV layer and a NaN there corrupts a ``.blend``.
3. **It is deterministic.**  Bit-identical input, bit-identical output.

The performance budget from section 3 -- a 100k-vertex Laplacian matvec under
10 ms -- is checked too, marked ``perf`` so it can be deselected.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from tests.conftest import import_or_skip


@pytest.fixture(scope="module")
def linalg():
    """The sparse-solver module under test."""
    return import_or_skip("unwrap_variations.core.linalg")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def dense_to_triplets(matrix: np.ndarray):
    """Return ``(rows, cols, vals, n)`` covering every entry of a dense matrix."""
    n = matrix.shape[0]
    rows, cols = np.nonzero(np.ones((n, n), dtype=bool))
    return rows, cols, matrix[rows, cols], n


def random_spd(n: int, seed: int) -> np.ndarray:
    """Return a random symmetric positive-definite matrix.

    ``M M^T`` is positive semi-definite for any ``M``; the ``n I`` shift makes it
    strictly positive definite and keeps the condition number sane enough that
    CG converges in far fewer than ``n`` iterations, which is what the tests
    want to measure -- correctness, not the convergence rate of a hard problem.
    """
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, n))
    return m @ m.T + n * np.eye(n)


def grid_laplacian(linalg, side: int, shift: float = 0.0):
    """Build the graph Laplacian of a triangulated ``side x side`` lattice.

    Six neighbours per interior node, matching a real triangle mesh's sparsity
    rather than a 5-point stencil's, so the timing test measures something the
    flattener would actually assemble.
    """
    n = side * side
    idx = np.arange(n).reshape(side, side)
    ei = np.concatenate(
        [idx[:, :-1].ravel(), idx[:-1, :].ravel(), idx[:-1, :-1].ravel()]
    )
    ej = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel(), idx[1:, 1:].ravel()])
    w = np.ones(ei.size)
    rows = [ei, ej, ei, ej]
    cols = [ej, ei, ei, ej]
    vals = [-w, -w, w, w]
    if shift:
        diag = np.arange(n)
        rows.append(diag)
        cols.append(diag)
        vals.append(np.full(n, shift))
    return linalg.SparseSym(
        np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), n
    )


# ---------------------------------------------------------------------------
# SparseSym
# ---------------------------------------------------------------------------


def test_matvec_matches_dense(linalg):
    """``A @ x`` agrees with the dense product on a random symmetric matrix."""
    rng = np.random.default_rng(0)
    dense = random_spd(40, 1)
    op = linalg.SparseSym(*dense_to_triplets(dense))
    x = rng.normal(size=40)
    assert np.allclose(op.matvec(x), dense @ x, atol=1e-12)


def test_matvec_multi_rhs_matches_column_by_column(linalg):
    """A ``(n, k)`` product equals ``k`` independent ``(n,)`` products."""
    dense = random_spd(30, 2)
    op = linalg.SparseSym(*dense_to_triplets(dense))
    block = np.random.default_rng(3).normal(size=(30, 4))
    together = op.matvec(block)
    assert together.shape == (30, 4)
    for col in range(4):
        assert np.array_equal(together[:, col], op.matvec(block[:, col]))


def test_duplicate_triplets_are_summed(linalg):
    """Repeating a coordinate adds its values rather than replacing them."""
    op = linalg.SparseSym([0, 0, 1], [0, 0, 1], [1.5, 2.5, 7.0], 2)
    assert op.nnz == 2
    assert op.diagonal() == pytest.approx([4.0, 7.0])


def test_out_of_range_and_non_finite_triplets_are_dropped(linalg):
    """A bad triplet is discarded and counted, never stored.

    One NaN in the triplets would otherwise turn every matvec into a vector of
    NaNs with no way for the caller to see where it came from.
    """
    op = linalg.SparseSym([0, 5, -1, 1], [0, 0, 0, 1], [2.0, 1.0, 1.0, np.nan], 2)
    assert op.dropped == 3
    assert op.nnz == 1
    assert np.isfinite(op.matvec(np.ones(2))).all()


def test_diagonal_ignores_off_diagonal_entries(linalg):
    """``diagonal()`` reads only ``i == j``."""
    op = linalg.SparseSym([0, 0, 1, 1], [0, 1, 0, 1], [3.0, -1.0, -1.0, 5.0], 2)
    assert op.diagonal() == pytest.approx([3.0, 5.0])


def test_symmetry_error_detects_an_unsymmetric_assembly(linalg):
    """Symmetry is the caller's contract, so it is measurable, not enforced."""
    good = linalg.SparseSym([0, 1], [1, 0], [2.0, 2.0], 2)
    bad = linalg.SparseSym([0, 1], [1, 0], [2.0, 3.0], 2)
    assert good.symmetry_error() == 0.0
    assert bad.symmetry_error() == pytest.approx(1.0)


def test_symmetry_error_is_correct_for_an_unsorted_pattern(linalg):
    """``symmetry_error`` must not depend on the order the triplets arrived in.

    ``__init__`` coalesces through ``np.unique`` and therefore hands back a
    pattern sorted by ``(row, col)``; ``from_coalesced`` -- the constructor
    ``_tutte`` and ``_build_preconditioner`` actually use -- does not sort at
    all.  ``symmetry_error`` looked the mirror entry up with ``searchsorted``,
    which silently returns nonsense on an unsorted key array, so the one
    diagnostic offered for the one property the class refuses to enforce raised
    a false alarm on 5 of the 6 orderings of an exactly symmetric 2x2.
    """
    rows, cols, vals = np.array([0, 1, 1]), np.array([1, 0, 1]), np.array([2.0, 2.0, 5.0])
    for perm in itertools.permutations(range(3)):
        idx = list(perm)
        op = linalg.SparseSym.from_coalesced(rows[idx], cols[idx], vals[idx], 2)
        assert np.allclose(op.to_dense(), op.to_dense().T), "fixture must be symmetric"
        assert op.symmetry_error() == 0.0, "false alarm for ordering %s" % (perm,)

    skew = linalg.SparseSym.from_coalesced(
        np.array([1, 0]), np.array([0, 1]), np.array([3.0, 2.0]), 2
    )
    assert skew.symmetry_error() == pytest.approx(1.0)


def test_diagonal_does_not_hand_out_its_cache(linalg):
    """The operator is documented immutable, so its diagonal cannot be edited.

    ``diagonal()`` memoises and returned the cached array itself, so a caller
    that scaled the result in place -- the natural way to build a preconditioner
    -- silently rewrote the operator's idea of its own diagonal for every later
    caller.
    """
    op = linalg.SparseSym([0, 1], [0, 1], [3.0, 5.0], 2)
    first = op.diagonal()
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0] = 1234.0
    assert op.diagonal()[0] == 3.0


def test_with_values_keeps_the_pattern(linalg):
    """The fast reassembly path reuses the indices and swaps the values."""
    op = linalg.SparseSym([0, 1, 0], [0, 1, 0], [1.0, 2.0, 3.0], 2)
    twice = op.with_values(op.values * 2.0)
    assert np.array_equal(twice.rows, op.rows)
    assert np.array_equal(twice.matvec(np.ones(2)), 2.0 * op.matvec(np.ones(2)))


def test_with_values_keeps_the_dropped_count(linalg):
    """Reassembling with new values does not un-drop the triplets already lost."""
    op = linalg.SparseSym([0, 1, 99, -3], [0, 1, 0, 0], [1.0, 1.0, 5.0, 5.0], 2)
    assert op.dropped == 2
    assert op.with_values(op.values * 2.0).dropped == 2


def test_empty_matrix_is_usable(linalg):
    """A zero-dimension operator answers instead of raising."""
    op = linalg.SparseSym([], [], [], 0)
    assert op.nnz == 0
    assert op.matvec(np.zeros(0)).shape == (0,)


def test_matvec_rejects_a_shape_mismatch(linalg):
    """A wrong-shaped operand is a programming error and is reported loudly.

    The never-raise guarantee covers *numerical* failure -- a singular system,
    an indefinite one.  A caller handing a 5-vector to a 3x3 operator has a bug,
    and silently padding it would hide the bug and return a plausible answer.
    """
    op = linalg.SparseSym([0], [0], [1.0], 3)
    with pytest.raises(ValueError):
        op.matvec(np.ones(5))


# ---------------------------------------------------------------------------
# pcg: correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_pcg_matches_dense_solve_on_random_spd(linalg, seed):
    """The whole point: CG must reproduce ``np.linalg.solve`` on an SPD system."""
    n = 48
    dense = random_spd(n, seed)
    op = linalg.SparseSym(*dense_to_triplets(dense))
    b = np.random.default_rng(seed + 100).normal(size=n)
    got, iters, residual = linalg.pcg(op, b, tol=1e-13, maxiter=4 * n)
    assert residual < 1e-11
    assert iters <= 4 * n
    assert np.allclose(got, np.linalg.solve(dense, b), atol=1e-8)


def test_pcg_solves_all_right_hand_sides_together(linalg):
    """``u`` and ``v`` solved as one ``(n, 2)`` block match two separate solves."""
    n = 40
    dense = random_spd(n, 9)
    op = linalg.SparseSym(*dense_to_triplets(dense))
    b = np.random.default_rng(11).normal(size=(n, 2))
    block, _, residual = linalg.pcg(op, b, tol=1e-13, maxiter=4 * n)
    assert residual < 1e-11
    assert block.shape == (n, 2)
    assert np.allclose(block, np.linalg.solve(dense, b), atol=1e-8)


def test_pcg_preserves_the_input_shape(linalg):
    """A 1-D right-hand side comes back 1-D, a 2-D one 2-D."""
    op = linalg.SparseSym(*dense_to_triplets(random_spd(12, 4)))
    assert linalg.pcg(op, np.ones(12))[0].shape == (12,)
    assert linalg.pcg(op, np.ones((12, 3)))[0].shape == (12, 3)


def test_pcg_on_a_laplacian_with_an_analytic_solution(linalg):
    """A 1-D Dirichlet Laplacian's solution is the straight line through its ends.

    ``-u'' = 0`` with ``u(0) = 0`` and ``u(1) = 1`` is exactly linear, and its
    discrete analogue on a uniform path graph is exactly linear too -- no
    discretisation error at all, so the check is against the true answer rather
    than against another approximation.
    """
    n = 25
    rows, cols, vals = [], [], []
    for i in range(n - 1):
        rows += [i, i + 1, i, i + 1]
        cols += [i, i + 1, i + 1, i]
        vals += [1.0, 1.0, -1.0, -1.0]
    # Dirichlet on both ends: interior nodes 1..n-2 solved, ends pinned.
    interior = np.arange(1, n - 1)
    remap = np.full(n, -1)
    remap[interior] = np.arange(interior.size)
    rows, cols, vals = np.array(rows), np.array(cols), np.array(vals)
    keep = (remap[rows] >= 0) & (remap[cols] >= 0)
    op = linalg.SparseSym(remap[rows[keep]], remap[cols[keep]], vals[keep], interior.size)
    rhs = np.zeros(interior.size)
    rhs[-1] = 1.0  # coupling to the pinned u(1) = 1
    got, _, residual = linalg.pcg(op, rhs, tol=1e-14, maxiter=500)
    assert residual < 1e-12
    assert np.allclose(got, np.linspace(0, 1, n)[1:-1], atol=1e-9)


def test_pcg_converges_on_a_consistent_singular_system(linalg):
    """A pure-Neumann Laplacian is singular but solvable when ``b`` is balanced.

    CG handles that case -- the iterate wanders in the null space, the residual
    still converges -- and the solver must not mistake it for a breakdown.
    """
    op = grid_laplacian(linalg, 12)
    rng = np.random.default_rng(5)
    b = rng.normal(size=op.n)
    b -= b.mean()  # orthogonal to the constant null vector
    got, _, residual = linalg.pcg(op, b, tol=1e-11, maxiter=4000)
    assert residual < 1e-9
    assert np.isfinite(got).all()


# ---------------------------------------------------------------------------
# pcg: the failure modes it must survive
# ---------------------------------------------------------------------------


def test_pcg_solves_around_a_zero_row(linalg):
    """``diag(1, 0, 0)`` with ``b = (1, 1, 1)`` gives ``x_0 = 1`` and says so.

    An all-zero row states ``0 = b_i``.  It constrains nothing and, by symmetry,
    is constrained by nothing, so the reachable part of the system still has an
    answer.  Leaving the row in the iteration instead makes CG's first step
    *increase* the residual and the solver hands back zeros -- the correct
    ``x_0 = 1`` thrown away.  The reported residual keeps the unsatisfiable part
    so the caller can still see the system was inconsistent.
    """
    op = linalg.SparseSym([0, 1], [0, 1], [1.0, 0.0], 3)
    got, _, residual = linalg.pcg(op, np.array([1.0, 1.0, 1.0]))
    assert np.isfinite(got).all()
    assert got[0] == pytest.approx(1.0)
    assert residual == pytest.approx(np.sqrt(2.0 / 3.0), rel=1e-9)


def test_pcg_on_a_structurally_singular_system_does_not_raise(linalg):
    """A rank-deficient system returns the best iterate, finite, with a residual."""
    op = linalg.SparseSym([0, 0, 1, 1], [0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0], 2)
    got, _, residual = linalg.pcg(op, np.array([1.0, -1.0]))
    assert np.isfinite(got).all()
    assert np.isfinite(residual)


def test_pcg_on_an_indefinite_system_detects_breakdown(linalg):
    """``p^T A p <= 0`` means CG's step length is meaningless; freeze, do not fail."""
    op = linalg.SparseSym(np.arange(5), np.arange(5), [1.0, -1.0, 2.0, -3.0, 1.0], 5)
    got, _, residual = linalg.pcg(op, np.ones(5))
    assert np.isfinite(got).all()
    assert np.isfinite(residual)


def test_pcg_never_returns_nan_for_a_nan_right_hand_side(linalg):
    """A poisoned column reports ``inf`` residual and a finite iterate.

    A NaN escaping here becomes a NaN UV layer and a corrupted ``.blend``, so
    the honest answer is "I could not solve this", loudly, in the residual.
    """
    op = linalg.SparseSym(*dense_to_triplets(random_spd(6, 8)))
    b = np.ones((6, 2))
    b[2, 0] = np.nan
    got, _, residual = linalg.pcg(op, b)
    assert np.isfinite(got).all()
    assert residual == np.inf


def test_pcg_tolerates_a_non_finite_initial_guess(linalg):
    """A NaN in ``x0`` is neutralised rather than propagated."""
    dense = random_spd(10, 12)
    op = linalg.SparseSym(*dense_to_triplets(dense))
    b = np.ones(10)
    x0 = np.full(10, np.nan)
    got, _, residual = linalg.pcg(op, b, x0=x0, tol=1e-12, maxiter=200)
    assert np.allclose(got, np.linalg.solve(dense, b), atol=1e-8)
    assert residual < 1e-10


def test_pcg_on_an_empty_system(linalg):
    """Zero unknowns is an answer, not an error."""
    op = linalg.SparseSym([], [], [], 0)
    got, iters, residual = linalg.pcg(op, np.zeros(0))
    assert got.shape == (0,)
    assert iters == 0
    assert residual == 0.0


def test_pcg_with_a_zero_iteration_budget_returns_the_initial_guess(linalg):
    """``maxiter=0`` must still return something finite and honest."""
    op = linalg.SparseSym(*dense_to_triplets(random_spd(8, 2)))
    got, iters, residual = linalg.pcg(op, np.ones(8), maxiter=0)
    assert iters == 0
    assert np.array_equal(got, np.zeros(8))
    assert residual == pytest.approx(1.0)


def test_pcg_unpreconditioned_still_solves(linalg):
    """``precond="none"`` is a diagnostic switch, not a broken path."""
    dense = random_spd(20, 7)
    op = linalg.SparseSym(*dense_to_triplets(dense))
    b = np.ones(20)
    got, _, residual = linalg.pcg(op, b, precond="none", tol=1e-13, maxiter=400)
    assert residual < 1e-11
    assert np.allclose(got, np.linalg.solve(dense, b), atol=1e-8)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_pcg_is_bit_identical_across_runs(linalg, seed):
    """Same arrays in, same bits out -- required by ORGANIC.md section 3."""
    op = linalg.SparseSym(*dense_to_triplets(random_spd(32, seed)))
    b = np.random.default_rng(seed).normal(size=(32, 2))
    first = linalg.pcg(op, b, tol=1e-12, maxiter=200)
    second = linalg.pcg(op, b, tol=1e-12, maxiter=200)
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_matvec_is_bit_identical_across_runs(linalg):
    """The operator itself is deterministic, not just the solver."""
    op = grid_laplacian(linalg, 20)
    x = np.random.default_rng(1).normal(size=op.n)
    assert np.array_equal(op.matvec(x), op.matvec(x))


# ---------------------------------------------------------------------------
# Performance budgets (docs/ORGANIC.md section 3)
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.slow
def test_matvec_budget_at_100k_vertices(linalg):
    """A 100k-vertex mesh Laplacian matvec must stay under 10 ms.

    The budget is what makes the descent affordable: it runs a handful of
    matvecs per iteration and tens of iterations per chart, so a 5x miss here is
    a 5x miss everywhere.
    """
    op = grid_laplacian(linalg, 317)
    assert op.n >= 100_000
    x = np.random.default_rng(0).normal(size=op.n)
    op.matvec(x)
    started = time.perf_counter()
    for _ in range(5):
        op.matvec(x)
    elapsed = (time.perf_counter() - started) / 5.0
    assert elapsed < 0.010, "matvec took %.2f ms at n=%d, budget 10 ms" % (
        elapsed * 1000.0,
        op.n,
    )


@pytest.mark.perf
@pytest.mark.slow
def test_pcg_cost_per_iteration_at_200k_unknowns(linalg):
    """Measure the 200k-unknown cost the honest way: per iteration, on a Laplacian.

    **This test replaces a misleading one, and the story is worth keeping.**  It
    used to solve a Laplacian *shifted by* ``1.0 * I`` to convergence and report
    the total -- 35 iterations, 0.91 s.  That number is real but it is not a
    Laplacian: the shift makes the system strongly diagonally dominant, and it
    flatters the solver by roughly 70-100x.  The same grid without the shift,
    which is what ``docs/ORGANIC.md`` section 3 actually specifies, needs about
    2900 iterations and a minute or more.

    Total time to convergence on an unpreconditioned Poisson problem is a
    property of the *problem's* conditioning, not of this module, and asserting
    it would either be vacuous or flaky.  Cost per iteration is the figure that
    belongs to the implementation, so that is what is measured and bounded.
    """
    op = grid_laplacian(linalg, 448)
    assert op.n >= 200_000
    b = np.random.default_rng(2).normal(size=(op.n, 2))
    b -= b.mean(axis=0)
    budget = 100
    started = time.perf_counter()
    got, iters, residual = linalg.pcg(op, b, tol=0.0, maxiter=budget)
    elapsed = time.perf_counter() - started
    assert iters == budget
    assert np.isfinite(got).all()
    assert np.isfinite(residual)
    per_iteration = elapsed / budget
    assert per_iteration < 0.060, "%.1f ms per CG iteration at n=%d" % (
        per_iteration * 1000.0,
        op.n,
    )


@pytest.mark.perf
@pytest.mark.slow
def test_pcg_converges_on_a_well_conditioned_200k_system(linalg):
    """A diagonally dominant 200k system converges quickly -- and is labelled as such.

    Kept separate from the Laplacian budget above precisely so the two numbers
    cannot be confused.  The ``+1.0 I`` shift is the whole reason this converges
    in tens of iterations rather than thousands; quoting it as "a 200k Laplacian
    solve" would be wrong by two orders of magnitude.
    """
    op = grid_laplacian(linalg, 448, shift=1.0)
    b = np.random.default_rng(2).normal(size=(op.n, 2))
    got, iters, residual = linalg.pcg(op, b, tol=1e-10, maxiter=4000)
    assert residual < 1e-9, "did not converge: residual %.3g after %d iters" % (
        residual,
        iters,
    )
    assert np.isfinite(got).all()
