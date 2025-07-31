from __future__ import annotations
import torch
import math
from collections.abc import Sequence
from typing import Literal
from ..typing import VectorLike, Tensor3, Vector
from .core import Strategy, DEFAULT_STRATEGY
from .core import _destructive_svd, _left_orth_2site, _right_orth_2site

#
# Type of LAPACK driver used for solving singular value decompositions.
# The "gesdd" algorithm is the default in Python and is faster, but it
# may produced wrong results, specially in ill-conditioned matrices.
#
SVD_LAPACK_DRIVER: Literal["gesvd", "gesdd"] = "gesvd"


def _schmidt_weights(A: Tensor3) -> Vector:
    d1, d2, d3 = A.shape
    # PyTorch's SVD function
    _, s, _ = torch.linalg.svd(
        A.reshape(d1 * d2, d3),
        full_matrices=False
    )
    s = s * s
    s = s / torch.sum(s)
    return s


def _vector2mps(
    state: VectorLike,
    dimensions: Sequence[int],
    strategy: Strategy = DEFAULT_STRATEGY,
    normalize: bool = True,
    center: int = -1,
) -> tuple[list[Tensor3], float]:
    """Construct a list of tensors for an MPS that approximates the state ψ
    represented as a complex vector in a Hilbert space.

    Parameters
    ----------
    ψ
        -- wavefunction with \\prod_i dimensions[i] elements
    dimensions -- list of dimensions of the Hilbert spaces that build ψ
    tolerance -- truncation criterion for dropping Schmidt numbers
    normalize -- boolean to determine if the MPS is normalized
    """
    ψ: torch.Tensor = torch.as_tensor(state).clone().reshape(1, -1, 1)
    L = len(dimensions)
    if math.prod(dimensions) != ψ.numel():
        raise Exception("Wrong dimensions specified when converting a vector to MPS")
    output = [ψ] * L
    if center < 0:
        center = L + center
    if center < 0 or center >= L:
        raise Exception("Invalid value of center in _vector2mps")
    err: float = 0.0
    for i in range(center):
        output[i], ψ, new_err = _left_orth_2site(
            ψ.reshape(ψ.shape[0], dimensions[i], -1, ψ.shape[-1]), strategy
        )
        err += new_err
    for i in range(L - 1, center, -1):
        ψ, output[i], new_err = _right_orth_2site(
            ψ.reshape(ψ.shape[0], -1, dimensions[i], ψ.shape[-1]), strategy
        )
        err += new_err
    if normalize:
        N = torch.linalg.norm(ψ.reshape(-1))
        ψ = ψ / N
        err /= float(N)
    output[center] = ψ
    return output, err


__all__ = ["_destructive_svd", "_schmidt_weights", "_vector2mps"]