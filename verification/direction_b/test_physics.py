import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import svdvals
from scipy.sparse.linalg import LinearOperator

from experiments.direction_b.physics import Helmholtz, solve


def test_rectangular_stencil_and_hermitian_adjoint():
    model = np.arange(1.0, 7.0).reshape(2, 3)
    op = Helmholtz(model, spacing=0.5, frequency=1 / np.pi, damping=0.2)
    expected = np.array([
        [3, -1, 0, -1, 0, 0],
        [-1, 2, -1, 0, -1, 0],
        [0, -1, 1, 0, 0, -1],
        [-1, 0, 0, 0, -1, 0],
        [0, -1, 0, -1, -1, -1],
        [0, 0, -1, 0, -1, -2],
    ]) - 0.2j * np.eye(6)
    assert op.shape == (2, 3)
    assert sparse.isspmatrix_csr(op.A) and sparse.isspmatrix_csr(op.AH)
    assert op.A.dtype == np.complex128
    np.testing.assert_allclose(op.A.toarray(), expected)
    np.testing.assert_allclose(op.AH.toarray(), expected.conj().T)


@pytest.mark.parametrize("damping", [0.03, 0.5])
def test_inverse_norm_bound_including_resonance(damping):
    # The single-cell Laplacian is 4. At c*m=4 the bound is attained.
    frequency = 1 / np.pi
    resonant = Helmholtz(np.ones((1, 1)), 1, frequency, damping)
    assert resonant.K == pytest.approx(1 / svdvals(resonant.A.toarray())[-1])
    model = np.array([[0.8, 0.7], [1.3, 1.1], [0.9, 1.5]])
    op = Helmholtz(model, 1, frequency, damping)
    assert 1 / svdvals(op.A.toarray())[-1] <= op.K * (1 + 1e-13)
    assert op.K == pytest.approx(1 / damping)


def test_boundary_sponge_is_nonnormal_with_fixed_model_derivative_and_inverse_bound():
    model = np.linspace(0.8, 1.2, 64).reshape(8, 8)
    args = dict(spacing=0.5, frequency=0.7, damping=0.1, sponge=0.9)
    op = Helmholtz(model, **args)
    # Moving down a central column, distance to the nearest edge is 0,1,2,3.
    np.testing.assert_allclose(op.damping_profile[:4, 4], [1.0, 0.5, 0.2, 0.1])
    np.testing.assert_allclose(-op.A.diagonal().imag.reshape(model.shape), op.damping_profile)
    assert op.damping_profile.min() == pytest.approx(0.1)
    matrix = op.A.toarray()
    adjoint = matrix.conj().T
    assert np.linalg.norm(matrix @ adjoint - adjoint @ matrix) > 0.1
    assert 1 / svdvals(matrix)[-1] <= op.K == pytest.approx(10)
    dm = np.sin(np.arange(64)).reshape(8, 8) / 10
    perturbed = Helmholtz(model + dm, **args)
    np.testing.assert_allclose(
        (perturbed.A - op.A).toarray(), -op.coefficient * np.diag(dm.ravel()), atol=1e-14
    )


@pytest.mark.parametrize("sponge", [-0.1, np.nan, np.inf])
def test_sponge_requires_finite_nonnegative_strength(sponge):
    with pytest.raises(ValueError):
        Helmholtz(np.ones((4, 4)), 1, 0.4, 0.1, sponge=sponge)


def test_forward_tangent_and_receiver_misfit_gradient_by_finite_differences():
    rng = np.random.default_rng(34)
    model = rng.uniform(0.8, 1.4, (4, 5))
    direction = rng.uniform(-1, 1, model.shape)
    b = np.zeros(model.size, dtype=complex)
    b[6] = 1 + 0.4j
    receivers = [1, 3, 11, 18]
    data = rng.normal(size=4) + 1j * rng.normal(size=4)
    args = dict(spacing=0.3, frequency=0.8, damping=0.15)

    def evaluate(m):
        op = Helmholtz(m, **args)
        u = solve(op.A, b, 1e-13).x
        residual = u[receivers] - data
        return op, u, residual, 0.5 * np.vdot(residual, residual).real

    op, u, residual, _ = evaluate(model)
    h = 2e-5
    _, u_plus, _, loss_plus = evaluate(model + h * direction)
    _, u_minus, _, loss_minus = evaluate(model - h * direction)
    tangent = solve(op.A, op.coefficient * direction.ravel() * u, 1e-13).x
    np.testing.assert_allclose(tangent, (u_plus - u_minus) / (2 * h), rtol=2e-7)
    adjoint_source = np.zeros_like(b)
    adjoint_source[receivers] = residual
    adjoint = solve(op.AH, adjoint_source, 1e-13).x
    gradient = op.coefficient * (adjoint.conj() * u).real
    assert gradient @ direction.ravel() == pytest.approx(
        (loss_plus - loss_minus) / (2 * h), rel=2e-7
    )


def test_solve_reports_true_work_and_respects_supplied_initial_guess():
    matrix = np.array([[1 + 0.5j, -2.0], [0.3j, 2.0 - 0.1j]])
    exact = np.array([2.0 + 0.2j, -0.4j])
    rhs = matrix @ exact
    calls = []

    def multiply(v):
        calls.append(None)
        return matrix @ v

    operator = LinearOperator(matrix.shape, matvec=multiply, dtype=matrix.dtype)
    cold = solve(operator, rhs, 1e-12)
    assert cold.iterations == 2
    assert cold.matvecs == len(calls) and cold.matvecs >= cold.iterations + 1
    assert cold.seconds >= 0
    assert cold.x.dtype == np.complex128
    assert np.linalg.norm(rhs - matrix @ cold.x) <= 1e-12 * np.linalg.norm(rhs)
    calls.clear()
    warm = solve(operator, rhs, 1e-12, x0=exact)
    assert warm.iterations == 0
    assert warm.matvecs == len(calls)
    np.testing.assert_allclose(warm.x, exact)


def test_zero_rhs_ignores_nonzero_initial_guess_without_krylov_work():
    result = solve(sparse.eye(3), np.zeros(3), 1e-8, x0=np.ones(3))
    np.testing.assert_array_equal(result.x, np.zeros(3))
    assert result.x.dtype == np.complex128
    assert result.iterations == result.matvecs == 0


@pytest.mark.parametrize("magnitude", [1e-180, 1e180])
def test_relative_tolerance_is_independent_of_source_amplitude(magnitude):
    rhs = magnitude * np.array([1 - 0.2j, 1.2j])
    matrix = sparse.diags([2 + 1j, 3 - 0.5j])
    solution = solve(matrix, rhs, 1e-12).x
    np.testing.assert_allclose((matrix @ solution) / magnitude, rhs / magnitude, rtol=1e-12)


def test_inconsistent_system_raises_instead_of_reporting_convergence():
    with pytest.raises(RuntimeError):
        solve(sparse.csr_matrix((3, 3)), np.ones(3), 1e-8)


@pytest.mark.parametrize("model,spacing,frequency,damping", [
    (np.array([1.0]), 1, 1, 0.1),
    (np.array([[np.nan]]), 1, 1, 0.1),
    (np.array([[0.0]]), 1, 1, 0.1),
    (np.array([[1.0]]), 0, 1, 0.1),
    (np.array([[1.0]]), 1, -1, 0.1),
    (np.array([[1.0]]), 1, 1, 0),
])
def test_invalid_physical_parameters_are_rejected(model, spacing, frequency, damping):
    with pytest.raises(ValueError):
        Helmholtz(model, spacing, frequency, damping)


@pytest.mark.parametrize("rhs,tol,initial", [
    ([1.0], 1e-6, None),
    ([np.nan, 1.0], 1e-6, None),
    ([1.0, 1.0], 0, None),
    ([1.0, 1.0], 1e-6, [1.0]),
])
def test_invalid_solver_inputs_are_rejected(rhs, tol, initial):
    with pytest.raises(ValueError):
        solve(sparse.eye(2), rhs, tol, x0=initial)
