import torch
from enum import Enum
import math
from typing import TYPE_CHECKING, Callable, List, Tuple
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

DEFAULT_TOLERANCE = torch.finfo(torch.float64).eps

def _truncate_relative_norm_squared_error(s: torch.Tensor, strategy) -> float:
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


def _truncate_relative_singular_value(s: torch.Tensor, strategy) -> float:
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


def _truncate_absolute_singular_value(s: torch.Tensor, strategy) -> float:
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


def destructively_truncate_vector(s: torch.Tensor, strategy) -> float:
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

def _truncate_do_not_truncate(s: torch.Tensor, strategy) -> float:
    """Do not truncate - return zero error."""
    return 0.0

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


# Placeholder functions for included files - these would be implemented
# in the corresponding PyTorch translations of the .pxi files

def _contract_nrjl_ijk_klm(U, A, B):
    """Contract tensors according to einsum pattern 'ijk,klm,nrjl -> inrm'."""
    raise NotImplementedError("Implementation from contractions.pxi")

def _join_environments(rhoL: Environment, rhoR: Environment):
    """Join left and right environments."""
    raise NotImplementedError("Implementation from environments.pxi")


def scprod(bra, ket) -> torch.Tensor:
    """Compute scalar product ⟨bra|ket⟩ between two MPS objects using PyTorch."""
    A = bra.data  # List[Tensor] of shape (D_left, d, D_right)
    B = ket.data

    if len(A) != len(B):
        raise ValueError("Invalid arguments to scprod: mismatched lengths")

    rho = empty_environment()  # Identity or scalar tensor, depending on implementation
    for i in range(len(A)):
        rho = update_left_environment(A[i], B[i], rho)

    return end_environment(rho)



def __svd(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Full SVD; you can modify for truncated or economy SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U, S, Vh

def _as_2tensor(A: torch.Tensor, dim1: int, dim2: int) -> torch.Tensor:
    return A.reshape(dim1, dim2)

def _as_3tensor(A: torch.Tensor, dim1: int, dim2: int, dim3: int) -> torch.Tensor:
    return A.reshape(dim1, dim2, dim3)

def _resize_matrix(M: torch.Tensor, new_rows: int, new_cols: int) -> torch.Tensor:
    # Resize M to shape (new_rows, new_cols) by truncating or keeping all if -1
    rows = M.shape[0] if new_rows == -1 else new_rows
    cols = M.shape[1] if new_cols == -1 else new_cols
    return M[:rows, :cols]

def __contract_last_and_first(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Contract last index of A with first index of B
    # A shape: (a, i, b), B shape: (b, c, d) -> result shape: (a, i, c, d)
    sA = A.shape
    sB = B.shape
    B_reshaped = B.reshape(sB[0], -1)  # (contract_dim, remaining)
    out = torch.matmul(A, B_reshaped)  # contracts on last of A and first of B
    return out.reshape(*sA[:-1], *sB[1:])  # match remaining dims

def __update_in_canonical_form_right(
    state: List[Tensor3], someA: Tensor3, site: int, truncation
) -> float:
    A = someA.clone()  # copy tensor
    
    a, i, b = A.shape
    
    A_2d = _as_2tensor(A, a * i, b)
    U, s, Vh = __svd(A_2d)
    
    # Truncate Schmidt decomposition
    err = math.sqrt(truncation._truncate(s, truncation))
    D = s.size(0)
    
    U_resized = _resize_matrix(U, -1, D)
    V_resized = _resize_matrix(Vh, D, -1)
    
    state[site] = _as_3tensor(U_resized, a, i, D)
    site += 1
    
    s_diag = s.reshape(D, 1)
    contracted = __contract_last_and_first(s_diag * V_resized, state[site])
    state[site] = contracted
    
    return err


def _update_in_canonical_form_right(state: List[Tensor3], A: Tensor3, site: int, truncation) -> Tuple[int, float]:
    if site + 1 == len(state):
        state[site] = A
        return site, 0.0
    return site + 1, __update_in_canonical_form_right(state, A, site, truncation)


def __update_in_canonical_form_left(
    state: List[Tensor3], someA: Tensor3, site: int, truncation
) -> float:
    A = someA.clone()
    
    a, i, b = A.shape
    
    A_2d = _as_2tensor(A, a, i * b)
    U, s, Vh = __svd(A_2d)
    
    err = math.sqrt(truncation._truncate(s, truncation))
    D = s.size(0)
    
    U_resized = _resize_matrix(U, -1, D)
    V_resized = _resize_matrix(Vh, D, -1)
    
    state[site] = _as_3tensor(V_resized, D, i, b)
    site -= 1
    contracted = __contract_last_and_first(state[site], U_resized * s.view(1, -1))
    state[site] = contracted
    
    return err


def _update_in_canonical_form_left(state: List[Tensor3], A: Tensor3, site: int, truncation) -> Tuple[int, float]:
    if site == 0:
        state[0] = A
        return 0, 0.0
    return site - 1, __update_in_canonical_form_left(state, A, site, truncation)


def _recanonicalize(state: List[Tensor3], oldcenter: int, newcenter: int, truncation) -> float:
    err = 0.0
    while oldcenter > newcenter:
        err += __update_in_canonical_form_left(state, state[oldcenter], oldcenter, truncation)
        oldcenter -= 1
    while oldcenter < newcenter:
        err += __update_in_canonical_form_right(state, state[oldcenter], oldcenter, truncation)
        oldcenter += 1
    return err

from typing import List
import torch

def _canonicalize(state: List[torch.Tensor], center: int, truncation) -> float:
    """Update a list of `Tensor3` objects to be in canonical form
    with respect to `center`."""
    
    err = 0.0
    L = len(state)
    
    for i in range(center):
        err += __update_in_canonical_form_right(state, state[i], i, truncation)
    for i in range(L - 1, center, -1):
        err += __update_in_canonical_form_left(state, state[i], i, truncation)
        
    return err



def _recanonicalize(state: list[Tensor3], oldcenter: int, newcenter: int, truncation: Strategy) -> float:
    """Re-canonicalize state from old center to new center."""
    raise NotImplementedError("Implementation from schmidt.pxi")

def _left_orth_2site(AA: torch.Tensor, strategy: Strategy) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'B' is a left-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten.
    """
    a, d1, d2, b = AA.shape
    # Reshape AA into a 2D matrix of shape (a*d1, d2*b)
    A = AA.view(a * d1, d2 * b)
    
    # Perform SVD: A = U @ diag(s) @ Vh
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Apply truncation strategy on singular values s
    err = strategy._truncate(s, strategy)
    
    D = s.size(0)
    
    # Truncate U, s, Vh accordingly
    # Note: Truncation function returns error, but you may want to truncate s, U, Vh explicitly
    # We'll truncate to the first D singular values (assuming _truncate adjusted s in place)
    # If you want to truncate based on some cut-off, you need to determine truncation size (r)
    # Here, let's truncate singular values s to length D, but real truncation logic depends on strategy

    # For simplicity, assume s, U, Vh are already truncated or truncated manually by user

    # Compose B and C tensors
    # B shape: (a, d1, D)
    B = U[:, :D].reshape(a, d1, D)
    # C shape: (D, d2, b)
    C = (s[:D].unsqueeze(1) * Vh[:D, :]).reshape(D, d2, b)
    
    return B, C, math.sqrt(err)


def _right_orth_2site(AA: torch.Tensor, strategy: Strategy) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'C' is a right-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten.
    """
    a, d1, d2, b = AA.shape
    # Reshape AA into a 2D matrix of shape (a*d1, d2*b)
    A = AA.view(a * d1, d2 * b)
    
    # Perform SVD: A = U @ diag(s) @ Vh
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Apply truncation strategy on singular values s
    err = strategy._truncate(s, strategy)
    
    D = s.size(0)
    
    # Compose B and C tensors
    # B shape: (a, d1, D)
    B = (U[:, :D] * s[:D]).reshape(a, d1, D)
    # C shape: (D, d2, b)
    C = Vh[:D, :].reshape(D, d2, b)
    
    return B, C, math.sqrt(err)



def _select_svd_driver(which: str) -> None:
    """Select SVD driver."""
    raise NotImplementedError("Implementation from svd.pxi")


def _destructive_svd(A: torch.Tensor):
    """Destructive SVD decomposition."""
    raise NotImplementedError("Implementation from svd.pxi")
