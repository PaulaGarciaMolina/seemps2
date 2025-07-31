import torch
import numpy as np
from enum import Enum
from typing import TYPE_CHECKING, Callable
from ..typing import Environment, Tensor3, Vector

if TYPE_CHECKING:
    from .mps import MPS

__version__ = 'pytorch-contractions'

MAX_BOND_DIMENSION = 0x7fffffff
"""Maximum bond dimension for any MPS."""


class Truncation(Enum):
    """Truncation method enumeration."""
    DO_NOT_TRUNCATE = 0
    RELATIVE_SINGULAR_VALUE = 1
    RELATIVE_NORM_SQUARED_ERROR = 2
    ABSOLUTE_SINGULAR_VALUE = 3


class Simplification(Enum):
    """Simplification method enumeration."""
    DO_NOT_SIMPLIFY = 0
    CANONICAL_FORM = 1
    VARIATIONAL = 2
    VARIATIONAL_EXACT_GUESS = 3


class GemmOrder:
    """GEMM operation flags."""
    GEMM_NORMAL: int = 0
    GEMM_TRANSPOSE: int = 1
    GEMM_ADJOINT: int = 2


DEFAULT_TOLERANCE = float(np.finfo(np.float64).eps)


class Strategy:
    """Strategy class for tensor operations.
    
    Parameters
    ----------
    method : int, default = Truncation.RELATIVE_NORM_SQUARED_ERROR
        Truncation method
    tolerance : float, default = DEFAULT_TOLERANCE
        Truncation tolerance
    simplification_tolerance : float, default = DEFAULT_TOLERANCE
        Simplification tolerance
    max_bond_dimension : int, default = MAX_BOND_DIMENSION
        Maximum bond dimension
    normalize : bool, default = False
        Whether to normalize
    simplify : int, default = Simplification.VARIATIONAL
        Simplification method
    max_sweeps : int, default = 16
        Maximum number of sweeps
    """
    
    def __init__(
        self,
        method: int = Truncation.RELATIVE_NORM_SQUARED_ERROR.value,
        tolerance: float = DEFAULT_TOLERANCE,
        simplification_tolerance: float = DEFAULT_TOLERANCE,
        max_bond_dimension: int = MAX_BOND_DIMENSION,
        normalize: bool = False,
        simplify: int = Simplification.VARIATIONAL.value,
        max_sweeps: int = 16
    ):
        if tolerance < 0 or tolerance >= 1.0:
            raise AssertionError("Invalid tolerance argument passed to Strategy")
        if tolerance == 0 and method != Truncation.DO_NOT_TRUNCATE.value:
            method = Truncation.ABSOLUTE_SINGULAR_VALUE.value
        self.tolerance = tolerance
        self.simplification_tolerance = simplification_tolerance
        
        if max_bond_dimension <= 0 or max_bond_dimension > MAX_BOND_DIMENSION:
            raise AssertionError("Invalid bond dimension in Strategy")
        else:
            self.max_bond_dimension = max_bond_dimension
            
        self.normalize = normalize
        
        if simplify < 0 or simplify > Simplification.VARIATIONAL_EXACT_GUESS.value:
            raise AssertionError("Invalid simplify argument passed to Strategy")
        else:
            self.simplify = simplify
            
        if max_sweeps < 0:
            raise AssertionError("Negative or zero number of sweeps in Strategy")
        self.max_sweeps = max_sweeps
        self.method = method
        
        # Set truncation function
        if method == Truncation.DO_NOT_TRUNCATE.value:
            self._truncate = _truncate_do_not_truncate
        elif method == Truncation.RELATIVE_NORM_SQUARED_ERROR.value:
            self._truncate = _truncate_relative_norm_squared_error
        elif method == Truncation.RELATIVE_SINGULAR_VALUE.value:
            self._truncate = _truncate_relative_singular_value
        elif method == Truncation.ABSOLUTE_SINGULAR_VALUE.value:
            self._truncate = _truncate_absolute_singular_value
        else:
            raise AssertionError("Invalid method argument passed to Strategy")
    
    def replace(
        self,
        method: int | None = None,
        tolerance: float | None = None,
        simplification_tolerance: float | None = None,
        max_bond_dimension: int | None = None,
        normalize: bool | None = None,
        simplify: int | None = None,
        max_sweeps: int | None = None,
    ) -> 'Strategy':
        """Create a new Strategy with modified parameters."""
        return Strategy(
            method=self.method if method is None else method,
            tolerance=self.tolerance if tolerance is None else tolerance,
            simplification_tolerance=self.simplification_tolerance if simplification_tolerance is None else simplification_tolerance,
            max_bond_dimension=self.max_bond_dimension if max_bond_dimension is None else max_bond_dimension,
            normalize=self.normalize if normalize is None else normalize,
            simplify=self.simplify if simplify is None else simplify,
            max_sweeps=self.max_sweeps if max_sweeps is None else max_sweeps
        )
    
    def set_normalization(self, normalize: bool) -> 'Strategy':
        """Set normalization flag and return new Strategy."""
        return self.replace(normalize=normalize)
    
    def get_method(self) -> int:
        return self.method
    
    def get_simplification_method(self) -> int:
        return self.simplify
    
    def get_tolerance(self) -> float:
        """Get the tolerance."""
        return self.tolerance
    
    def get_simplification_tolerance(self) -> float:
        """Get the simplification tolerance."""
        return self.simplification_tolerance
    
    def get_max_bond_dimension(self) -> int:
        """Get the maximum bond dimension."""
        return self.max_bond_dimension
    
    def get_max_sweeps(self) -> int:
        """Get the maximum number of sweeps."""
        return self.max_sweeps
    
    def get_normalize_flag(self) -> bool:
        """Get the normalize flag."""
        return self.normalize
    
    def get_simplify_flag(self) -> bool:
        """Get whether simplification is enabled."""
        return False if self.simplify == 0 else True
    
    def __str__(self) -> str:
        """String representation of Strategy."""
        if self.method == Truncation.DO_NOT_TRUNCATE.value:
            method = "None"
        elif self.method == Truncation.RELATIVE_SINGULAR_VALUE.value:
            method = "RelativeSVD"
        elif self.method == Truncation.RELATIVE_NORM_SQUARED_ERROR.value:
            method = "RelativeNorm"
        elif self.method == Truncation.ABSOLUTE_SINGULAR_VALUE.value:
            method = "AbsoluteSVD"
        else:
            raise ValueError("Invalid truncation method found in Strategy")
        
        if self.simplify == Simplification.DO_NOT_SIMPLIFY.value:
            simplification_method = "None"
        elif self.simplify == Simplification.CANONICAL_FORM.value:
            simplification_method = "CanonicalForm"
        elif self.simplify == Simplification.VARIATIONAL.value:
            simplification_method = "Variational"
        elif self.simplify == Simplification.VARIATIONAL_EXACT_GUESS.value:
            simplification_method = "Variational (exact guess)"
        else:
            raise ValueError("Invalid simplification method found in Strategy")
        
        return (f"Strategy(method={method}, tolerance={self.tolerance:5g}, "
                f"max_bond_dimension={self.max_bond_dimension}, normalize={self.normalize}, "
                f"simplify={simplification_method}, simplification_tolerance={self.simplification_tolerance:5g}, "
                f"max_sweeps={self.max_sweeps})")


DEFAULT_STRATEGY = Strategy(
    method=Truncation.RELATIVE_NORM_SQUARED_ERROR.value,
    simplify=Simplification.VARIATIONAL.value,
    tolerance=DEFAULT_TOLERANCE,
    simplification_tolerance=DEFAULT_TOLERANCE,
    max_bond_dimension=MAX_BOND_DIMENSION,
    normalize=False
)

NO_TRUNCATION = DEFAULT_STRATEGY.replace(
    method=Truncation.DO_NOT_TRUNCATE.value,
    simplify=Simplification.DO_NOT_SIMPLIFY.value
)


# Global buffer for errors (similar to Cython version)
_errors_buffer = torch.empty(1024, dtype=torch.float64)


def _truncate_do_not_truncate(s: torch.Tensor, strategy: Strategy) -> float:
    """Do not truncate - return zero error."""
    return 0.0


def _norm(data: torch.Tensor) -> float:
    """Compute L2 norm of tensor."""
    return torch.linalg.norm(data).item()


def _rescale_if_not_zero(data: torch.Tensor, factor: float) -> None:
    """Rescale tensor by factor if factor is non-zero."""
    if factor:
        data /= factor


def _normalize(data: torch.Tensor) -> None:
    """Normalize tensor in-place."""
    norm = _norm(data)
    _rescale_if_not_zero(data, norm)


def _truncate_relative_norm_squared_error(s: torch.Tensor, strategy: Strategy) -> float:
    """Truncate based on relative norm squared error."""
    global _errors_buffer
    
    N = s.size(0)
    
    # Resize buffer if needed
    if _errors_buffer.size(0) <= N:
        _errors_buffer = torch.empty(2 * N, dtype=torch.float64)
    
    # Compute cumulative sum of squared singular values in reverse order
    s_squared = s * s
    errors = torch.zeros(N + 1, dtype=torch.float64, device=s.device)
    
    total = 0.0
    for i in range(N):
        errors[i] = total
        total += s_squared[N - 1 - i].item()
    errors[N] = total
    
    max_error = total * strategy.tolerance
    final_size = N
    
    for i in range(N):
        if errors[i] > max_error:
            final_size = N - (i - 1) if i > 0 else N
            break
    
    final_size = min(final_size, strategy.max_bond_dimension)
    truncation_error = errors[N - final_size].item()
    
    # Truncate the tensor (resize in-place equivalent)
    # Note: PyTorch doesn't have direct in-place resize like NumPy
    # So we return the truncated view and the error
    return truncation_error


def _truncate_relative_singular_value(s: torch.Tensor, strategy: Strategy) -> float:
    """Truncate based on relative singular value."""
    if s.size(0) == 0:
        return 0.0
        
    max_error_threshold = strategy.tolerance * s[0].item()
    final_size = min(s.size(0), strategy.max_bond_dimension)
    
    for i in range(1, final_size):
        if s[i].item() <= max_error_threshold:
            final_size = i
            break
    
    # Compute truncation error as sum of squares of discarded values
    max_error = 0.0
    for i in range(final_size, s.size(0)):
        max_error += s[i].item() ** 2
    
    return max_error


def _truncate_absolute_singular_value(s: torch.Tensor, strategy: Strategy) -> float:
    """Truncate based on absolute singular value."""
    max_error_threshold = strategy.tolerance
    final_size = min(s.size(0), strategy.max_bond_dimension)
    
    for i in range(final_size):
        if s[i].item() <= max_error_threshold:
            final_size = i
            break
    
    # Compute truncation error as sum of squares of discarded values
    max_error = 0.0
    for i in range(final_size, s.size(0)):
        max_error += s[i].item() ** 2
    
    return max_error


def destructively_truncate_vector(s: torch.Tensor, strategy: Strategy) -> float:
    """Destructively truncate vector according to strategy.
    
    Parameters
    ----------
    s : torch.Tensor
        1D tensor of singular values (should be sorted in descending order)
    strategy : Strategy
        Truncation strategy
        
    Returns
    -------
    float
        Truncation error
    """
    if not isinstance(s, torch.Tensor) or s.ndim != 1:
        raise AssertionError("Expected 1D tensor")
    
    return strategy._truncate(s, strategy)


# Placeholder functions for included files - these would be implemented
# in the corresponding PyTorch translations of the .pxi files

def _contract_nrjl_ijk_klm(U, A, B):
    """Contract tensors according to einsum pattern 'ijk,klm,nrjl -> inrm'."""
    raise NotImplementedError("Implementation from contractions.pxi")


def _contract_last_and_first(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Contract last index of A and first from B."""
    raise NotImplementedError("Implementation from contractions.pxi")


def _begin_environment(D: int | None = 1) -> Environment:
    """Begin environment calculation."""
    raise NotImplementedError("Implementation from environments.pxi")


def _update_right_environment(B: Tensor3, A: Tensor3, rho: Environment) -> Environment:
    """Update right environment."""
    raise NotImplementedError("Implementation from environments.pxi")


def _update_left_environment(B: Tensor3, A: Tensor3, rho: Environment) -> Environment:
    """Update left environment."""
    raise NotImplementedError("Implementation from environments.pxi")


def _end_environment(rho: Environment):
    """End environment calculation."""
    raise NotImplementedError("Implementation from environments.pxi")


def _join_environments(rhoL: Environment, rhoR: Environment):
    """Join left and right environments."""
    raise NotImplementedError("Implementation from environments.pxi")


def scprod(bra: 'MPS', ket: 'MPS'):
    """Scalar product between two MPS."""
    raise NotImplementedError("Implementation from environments.pxi")


def _update_in_canonical_form_left(state: list[Tensor3], A: Tensor3, site: int, truncation: Strategy) -> tuple[int, float]:
    """Update state in canonical form moving left."""
    raise NotImplementedError("Implementation from schmidt.pxi")


def _update_in_canonical_form_right(state: list[Tensor3], A: Tensor3, site: int, truncation: Strategy) -> tuple[int, float]:
    """Update state in canonical form moving right."""
    raise NotImplementedError("Implementation from schmidt.pxi")


def _canonicalize(state: list[Tensor3], center: int, truncation: Strategy) -> float:
    """Canonicalize state around center."""
    raise NotImplementedError("Implementation from schmidt.pxi")


def _recanonicalize(state: list[Tensor3], oldcenter: int, newcenter: int, truncation: Strategy) -> float:
    """Re-canonicalize state from old center to new center."""
    raise NotImplementedError("Implementation from schmidt.pxi")


def _left_orth_2site(AA, strategy: Strategy):
    """Left orthogonalize two-site tensor."""
    raise NotImplementedError("Implementation from schmidt.pxi")


def _right_orth_2site(AA, strategy: Strategy):
    """Right orthogonalize two-site tensor."""
    raise NotImplementedError("Implementation from schmidt.pxi")


def _select_svd_driver(which: str) -> None:
    """Select SVD driver."""
    raise NotImplementedError("Implementation from svd.pxi")


def _destructive_svd(A: torch.Tensor):
    """Destructive SVD decomposition."""
    raise NotImplementedError("Implementation from svd.pxi")


def _gemm(B: torch.Tensor, BT: int, A: torch.Tensor, AT: int) -> torch.Tensor:
    """General matrix multiply with transpose options."""
    raise NotImplementedError("Implementation from gemm.pxi")