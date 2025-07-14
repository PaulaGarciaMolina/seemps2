from __future__ import annotations
import numpy as np
from ..tools import make_logger
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, Strategy
from ..truncate.simplify import AntilinearForm
from ..mpo import MPO
from ..optimization.dmrg import QuadraticForm


def dmrg_solve(
    A: MPO,
    b: MPS,
    guess: MPS | None = None,
    maxiter: int = 20,
    atol: float = 0,
    rtol: float = 1e-5,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> tuple[MPS, float]:
    r"""Solve an inverse problem :math:`A x = b` for an MPO `A` and an MPS `b` using DMRG.

    Given the :class:`MPO` `A` and the :class:`MPS` `b`, use the DMRG
    method to estimate another MPS that solves the linear system of
    equations :math:`A \\psi = b`. Convergence is determined by the
    residual :math:`\\Vert{A \\psi - b}\\Vert` being smaller than `tol`.

    Parameters
    ----------
    A : MPO
        The linear operator that on the left-hand-side of the equation.
    b : MPS
        The state at the right-hand-side of the equation.
    guess : MPS, default = b
        An initial guess for the ground state.
    maxiter : int, default = 20
        Maximum number of steps of the DMRG. Each step is a sweep that runs
        over every pair of neighborin sites. Defaults to 20.
    atol, rtol : float
        Absolute and relative tolerance for the convergence of the algorithm.
        `norm(A@x - b) <= max(rtol * norm(b), atol)`. Defaults are
        `rtol=1e-5` and `atol=0`
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy to keep bond dimensions in check. Defaults to
        `DEFAULT_STRATEGY`, which is very strict.

    Returns
    -------
    MPS
        The unknown :math:`x`.
    float
        Residual :math:`\Vert{A x - b}\Vert`.
    """
    if maxiter < 1:
        raise Exception("maxiter cannot be zero or negative")
    if guess is None:
        guess = b.copy()
    tol = max(atol, rtol * b.norm())
    logger = make_logger()
    logger(f"DMRG solver initiated with maxiter={maxiter}, absolute tolerance={tol}")
    if not isinstance(guess, CanonicalMPS):
        guess = CanonicalMPS(guess, center=0)
    if guess.center == 0:
        direction = +1
        QF = QuadraticForm(A, guess, start=0)
        LF = AntilinearForm(guess, b, center=0)
    else:
        direction = -1
        QF = QuadraticForm(A, guess, start=A.size - 2)
        LF = AntilinearForm(guess, b, center=A.size - 2)
    strategy = strategy.replace(normalize=True)
    step: int = 0
    residual: float = np.inf
    message: str = f"Exceeded number of steps {maxiter}"
    for step in range(maxiter):
        if step:
            if direction > 0:
                for i in range(0, A.size - 1):
                    AB, info, local_residual = QF.solve(
                        i, LF.tensor2site(+1), atol, rtol
                    )
                    QF.update_2site_right(AB, i, strategy)
                    LF.update_right()
                    logger(
                        f"-> site={i}, error={local_residual}, converged={info == 0}"
                    )
                    if info:
                        message = "Local optimization with gmres() did not converge"
            else:
                for i in range(A.size - 2, -1, -1):
                    AB, info, local_residual = QF.solve(
                        i, LF.tensor2site(-1), atol, rtol
                    )
                    QF.update_2site_left(AB, i, strategy)
                    LF.update_left()
                    logger(
                        f"-> site={i}, error={local_residual}, converged={info == 0}"
                    )
                    if info:
                        message = "Local optimization with gmres() did not converge"
            direction = -direction

        # In principle, E is the exact eigenvalue. However, we have
        # truncated the eigenvector, which means that the computation of
        # the residual cannot use that value
        residual = (A @ QF.state - b).norm()
        logger(f"step={step}, error={residual}")
        if residual < tol:
            message = f"Algorithm converged below tolerance {tol}"
            break
    logger(f"DMRG finished with {step + 1} iterations:\nmessage = {message}")
    logger.close()
    return QF.state, abs(residual)
