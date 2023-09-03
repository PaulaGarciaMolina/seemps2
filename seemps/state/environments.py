import numpy as np
from ..typing import *


def begin_environment(χ: int = 1) -> Environment:
    """Initiate the computation of a left environment from two MPSLike. The bond
    dimension χ defaults to 1. Other values are used for states in canonical
    form that we know how to open and close."""
    return np.eye(χ, dtype=np.float64)


def update_left_environment(
    B: Tensor3, A: Tensor3, rho: Environment, operator: Optional[Operator] = None
) -> Environment:
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if operator is not None:
        # A = np.einsum("ji,aib->ajb", operator, A)
        A = np.matmul(operator, A)
    # np.einsum("ijk,li,ljk->nk", A, rho, B.conj())
    i, j, k = A.shape
    l, j, n = B.shape
    # np.einsum("li,ijk->ljk")
    rho = np.matmul(rho, A.reshape(i, j * k))
    # np.einsum("nlj,ljk->nk")
    return np.matmul(B.reshape(l * j, n).T.conj(), rho.reshape(l * j, k))


def update_right_environment(
    B: Tensor3, A: Tensor3, rho: Environment, operator: Optional[Operator] = None
) -> Environment:
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if operator is not None:
        # A = np.einsum("ji,aib->ajb", operator, A)
        A = np.matmul(operator, A)
    # np.einsum("ijk,kn,ljn->il", A, rho, B.conj())
    i, j, k = A.shape
    l, j, n = B.shape
    # np.einsum("ijk,kn->ijn", A, rho)
    rho = np.matmul(A.reshape(i * j, k), rho)
    return np.matmul(rho.reshape(i, j * n), B.reshape(l, j * n).T.conj())


def end_environment(ρ: Environment) -> Weight:
    """Extract the scalar product from the last environment."""
    return ρ[0, 0]


# TODO: Separate formats for left- and right- environments so that we
# can replace this with a simple np.dot(ρL.reshape(-1), ρR.reshape(-1))
# This involves ρR -> ρR.T with respect to current conventions
def join_environments(ρL: Environment, ρR: Environment) -> Weight:
    """Join left and right environments to produce a scalar."""
    # np.einsum("ij,ji", ρL, ρR)
    # return np.trace(np.dot(ρL, ρR))
    return np.dot(ρL.reshape(-1), ρR.T.reshape(-1))


def scprod(bra: MPSLike, ket: MPSLike) -> Weight:
    """Compute the scalar product between matrix product states
    :math:`\\langle\\xi|\\psi\\rangle`.

    Parameters
    ----------
    bra : MPS
        Matrix-product state for the bra :math:`\\xi`
    ket : MPS
        Matrix-product state for the ket :math:`\\psi`

    Returns
    -------
    float | complex
        Scalar product.
    """
    ρ: Environment = begin_environment()
    # TODO: Verify if the order of Ai and Bi matches being bra and ket
    # Add tests for that
    for Ai, Bi in zip(bra, ket):
        ρ = update_left_environment(Ai, Bi, ρ)
    return end_environment(ρ)
