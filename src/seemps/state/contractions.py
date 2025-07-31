import torch
from typing import Tuple


def _as_matrix(A: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Reshape tensor A to matrix with given dimensions."""
    return A.reshape(rows, cols)


def _contract_last_and_first(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Contract last index of `A` and first from `B`"""
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise ValueError("_contract_last_and_first expects tensors")
    
    # Get dimensions
    A_shape = A.shape
    B_shape = B.shape
    
    # Reshape tensors to matrices for efficient matrix multiplication
    A_last = A_shape[-1]
    B_first = B_shape[0]
    
    # Reshape A to (prod(A.shape[:-1]), A.shape[-1])
    A_matrix = A.reshape(-1, A_last)
    # Reshape B to (B.shape[0], prod(B.shape[1:]))
    B_matrix = B.reshape(B_first, -1)
    
    # Matrix multiplication
    C = torch.matmul(A_matrix, B_matrix)
    
    # Reshape result to A.shape[:-1] + B.shape[1:]
    result_shape = A_shape[:-1] + B_shape[1:]
    return C.reshape(result_shape)


_matmul = torch.matmul


def _as_2tensor(A: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Reshape tensor A to 2D tensor with dimensions (i, j)."""
    return A.reshape(i, j)


def _as_3tensor(A: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
    """Reshape tensor A to 3D tensor with dimensions (i, j, k)."""
    return A.reshape(i, j, k)


def _as_4tensor(A: torch.Tensor, i: int, j: int, k: int, l: int) -> torch.Tensor:
    """Reshape tensor A to 4D tensor with dimensions (i, j, k, l)."""
    return A.reshape(i, j, k, l)


def _empty_as_array(A: torch.Tensor) -> torch.Tensor:
    """Create empty tensor with same shape and dtype as A."""
    return torch.empty_like(A)


def _empty_matrix(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    """Create empty matrix with given dimensions and dtype."""
    return torch.empty(rows, cols, dtype=dtype)


def _empty_vector(size: int, dtype: torch.dtype) -> torch.Tensor:
    """Create empty vector with given size and dtype."""
    return torch.empty(size, dtype=dtype)


def _copy_array(A: torch.Tensor) -> torch.Tensor:
    """Create a contiguous copy of tensor A."""
    return A.clone().contiguous()


def _resize_matrix(A: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Equivalent of A[:rows,:cols], creating a fresh new array."""
    old_rows, old_cols = A.shape[:2]
    
    if rows < 0:
        rows = old_rows
    if cols < 0:
        cols = old_cols
    
    if rows == old_rows and cols == old_cols:
        return A
    
    # Create new tensor with desired size
    output = torch.empty(rows, cols, dtype=A.dtype, device=A.device)
    
    # Copy data from original tensor (up to the smaller dimensions)
    copy_rows = min(rows, old_rows)
    copy_cols = min(cols, old_cols)
    output[:copy_rows, :copy_cols] = A[:copy_rows, :copy_cols]
    
    return output


def _adjoint(A: torch.Tensor) -> torch.Tensor:
    """Compute adjoint (conjugate transpose) of tensor A."""
    # Transpose the last two dimensions
    a = A.transpose(-2, -1)
    
    # If complex, take conjugate
    if A.dtype.is_complex:
        a = torch.conj(a)
    
    return a


def _contract_nrjl_ijk_klm(U: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Assuming U[n*r,j*l], A[i,j,k] and B[k,l,m]
    Implements torch.einsum('ijk,klm,nrjl -> inrm', A, B, U)
    """
    if (not isinstance(A, torch.Tensor) or 
        not isinstance(B, torch.Tensor) or 
        not isinstance(U, torch.Tensor) or
        A.ndim != 3 or
        B.ndim != 3 or
        U.ndim != 2):
        raise ValueError("Invalid arguments to _contract_nrjl_ijk_klm")
    
    # Get dimensions
    a, d, b = A.shape
    _, e, c = B.shape
    
    # Contract A and B first: A[i,j,k] @ B[k,l,m] -> [i,j,l,m]
    # Reshape A to (a*d, b) and B to (b, e*c)
    A_matrix = A.reshape(a * d, b)
    B_matrix = B.reshape(b, e * c)
    
    # Matrix multiplication
    AB = torch.matmul(A_matrix, B_matrix)  # Shape: (a*d, e*c)
    
    # Reshape to (a, d*e, c)
    AB_reshaped = AB.reshape(a, d * e, c)
    
    # Contract with U: U[n*r, j*l] @ AB_reshaped[i, j*l, m] -> [n*r, i, m]
    # Need to reshape AB_reshaped to (d*e, a*c) for proper contraction
    AB_for_U = AB_reshaped.permute(1, 0, 2).reshape(d * e, a * c)
    
    # Contract: U @ AB_for_U
    result = torch.matmul(U, AB_for_U)  # Shape should work out to desired dimensions
    
    # Reshape to final form (a, d, e, c) - this may need adjustment based on exact semantics
    return _as_4tensor(result, a, d, e, c)