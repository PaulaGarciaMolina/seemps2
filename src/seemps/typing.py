from __future__ import annotations
import torch
from torch import Tensor
from typing import TypeAlias, Annotated, TypeVar, Union

Natural: TypeAlias = Annotated[int, ">=1"]

Weight: TypeAlias = float | complex
"""A real or complex number."""

Unitary: TypeAlias = Tensor
"""Unitary matrix in :class:`torch.Tensor` dense format."""

SparseOperator: TypeAlias = torch.Tensor  # PyTorch has sparse tensors, but format differs
"""An operator in sparse tensor format."""

Operator: TypeAlias = Tensor
"""An operator, either dense or sparse in :class:`torch.Tensor` format."""

DenseOperator: TypeAlias = Tensor
"""An operator in :class:`torch.Tensor` dense format."""

Vector: TypeAlias = Tensor
"""A one-dimensional :class:`torch.Tensor` representing a wavefunction."""

VectorLike: TypeAlias = Union[float, list[float], Tensor]
"""Any Python type that can be coerced to Vector type."""

Tensor3: TypeAlias = Tensor
""":class:`torch.Tensor` tensor with three indices."""

Tensor4: TypeAlias = Tensor
""":class:`torch.Tensor` tensor with four indices."""

Environment: TypeAlias = Tensor
"""Left or right environment represented as tensor."""

MPOEnvironment: TypeAlias = Tensor
"""Left or right environment of an MPS-MPO-MPS contraction."""

FloatOrArray = TypeVar("FloatOrArray", float, Tensor)

def to_dense_operator(O: Operator) -> DenseOperator:
    # If sparse, convert using PyTorch's sparse API
    if O.is_sparse:
        return O.to_dense()
    return O

__all__ = [
    "Tensor",
    "Weight",
    "Vector",
    "VectorLike",
    "Operator",
    "Unitary",
    "DenseOperator",
    "Tensor3",
    "Tensor4",
    "Environment",
    "MPOEnvironment",
    "to_dense_operator",
]
