from __future__ import annotations
import torch
from ..typing import Weight, Tensor3, Tensor4, MPOEnvironment
from .core import (
    _begin_environment,
    _update_left_environment,
    _update_right_environment,
    _end_environment,
    _join_environments,
    scprod,
)


def begin_mpo_environment() -> MPOEnvironment:
    """Initialize MPO environment."""
    return torch.ones((1, 1, 1), dtype=torch.float64)


def update_left_mpo_environment(
    rho: MPOEnvironment, A: Tensor3, O: Tensor4, B: Tensor3
) -> MPOEnvironment:
    """Update left MPO environment.
    
    Implements: einsum("acb,ajd,cjie,bif->def", rho, A, O, B)
    """
    # bif,acb->ifac
    aux = torch.tensordot(B, rho, dims=([0], [2]))
    # ifac,cjie->faje
    aux = torch.tensordot(aux, O, dims=([0, 3], [2, 0]))
    # faje,ajd-> def
    aux = torch.tensordot(aux, A, dims=([1, 2], [0, 1])).permute(2, 1, 0)
    return aux


def update_right_mpo_environment(
    rho: MPOEnvironment, A: Tensor3, O: Tensor4, B: Tensor3
) -> MPOEnvironment:
    """Update right MPO environment.
    
    Implements: einsum("def,ajd,cjie,bif->acb", rho, A, O, B)
    """
    # ajd,def->ajef
    aux = torch.tensordot(A, rho, dims=([2], [0]))
    # ajef,cjie->afci
    aux = torch.tensordot(aux, O, dims=([1, 2], [1, 3]))
    # afci,bif->acb
    aux = torch.tensordot(aux, B, dims=([1, 3], [2, 1]))
    return aux


def end_mpo_environment(ρ: MPOEnvironment) -> Weight:
    """Extract the scalar product from the last environment."""
    return ρ[0, 0, 0].item()


def join_mpo_environments(left: MPOEnvironment, right: MPOEnvironment) -> Weight:
    """Join two MPO environments by computing their dot product."""
    return torch.dot(left.reshape(-1), right.reshape(-1)).item()


__all__ = [
    "_begin_environment",
    "_update_left_environment",
    "_update_right_environment",
    "_end_environment",
    "_join_environments",
    "scprod",
    "begin_mpo_environment",
    "update_left_mpo_environment",
    "update_right_mpo_environment",
    "end_mpo_environment",
    "join_mpo_environments",
]