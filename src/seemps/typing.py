

from __future__ import annotations import numpy as np from math import sqrt from collections.abc import Iterable, Sequence from typing import cast from ..tools import InvalidOperation from ..typing import Weight, Vector, Tensor3 from .environments import scprod class MPSSum: """Clas

pasted

I have a python library in numpy that I need to change to torch. I'll send you code snippets to do the necessary changes. Please, don't change anything else, only torch compatibility. Suppose that imported functions are already okay. The first file is mpssum.py:

Edit
I'll help you convert this NumPy-based code to PyTorch. Looking at the mpssum.py file, I need to replace NumPy operations with their PyTorch equivalents while maintaining the same functionality.


mpssum.py - PyTorch Version
Code 
I've converted the NumPy code to PyTorch. Here are the key changes I made:

Import change: Replaced import numpy as np with import torch
Tensor concatenation:
Changed np.concatenate([...], axis=2) to torch.cat([...], dim=2)
Changed np.concatenate(As, axis=0) to torch.cat(As, dim=0)
Zero tensor creation:
Changed np.zeros((DL, d, DR), dtype=type(w)) to torch.zeros((DL, d, DR), dtype=type(w))
Complex conjugate:
Updated the conj() method to handle both torch tensors and regular Python numbers
Used torch.conj(w) for tensors and complex(w).conjugate() for regular numbers
All other functionality remains the same, as the operations like list manipulations, arithmetic operations on weights, and the overall logic don't need to change for PyTorch compatibility. The tensor operations are now using PyTorch's API instead of NumPy's.





from cpython cimport ( PyList_Check, PyList_GET_SIZE, PyList_SetItem, PyList_SetItem, PyList_GET_ITEM, PyTuple_GET_ITEM, Py_INCREF, ) cdef void state_set(list state, Py_ssize_t n, cnp.ndarray A) noexcept: Py_INCREF(A) PyList_SetItem(state, n, A)

pasted

Now I have schmidt.py: from future import annotations import numpy as np import math from collections.abc import Sequence from numpy.typing import NDArray from typing import Literal from ..typing import VectorLike, Tensor3, Vector from .core import Strategy, DEFAULT_STRATEGY from scipy.linalg import svd as _scipy_svd from .core import _destructive_svd, _left_orth_2site, _right_orth_2site  # # Type of LAPACK driver used for solving singular value decompositions. # The "gesdd" algorithm is the default in Python and is faster, but it # may produced wrong results, specially in ill-conditioned matrices. # SVD_LAPACK_DRIVER: Literal["gesvd", "gesdd"] = "gesvd"   def _schmidt_weights(A: Tensor3) -> Vector:     d1, d2, d3 = A.shape     s: Vector = _scipy_svd(         A.reshape(d1 * d2, d3),         full_matrices=False,         compute_uv=False,         check_finite=False,         lapack_driver=SVD_LAPACK_DRIVER,     )     s *= s     s /= np.sum(s)     return s   def _vector2mps(     state: VectorLike,     dimensions: Sequence[int],     strategy: Strategy = DEFAULT_STRATEGY,     normalize: bool = True,     center: int = -1, ) -> tuple[list[Tensor3], float]:     """Construct a list of tensors for an MPS that approximates the state ψ     represented as a complex vector in a Hilbert space.      Parameters     ----------     ψ         -- wavefunction with \\prod_i dimensions[i] elements     dimensions -- list of dimensions of the Hilbert spaces that build ψ     tolerance -- truncation criterion for dropping Schmidt numbers     normalize -- boolean to determine if the MPS is normalized     """     ψ: NDArray = np.asarray(state).copy().reshape(1, -1, 1)     L = len(dimensions)     if math.prod(dimensions) != ψ.size:         raise Exception("Wrong dimensions specified when converting a vector to MPS")     output = [ψ] * L     if center < 0:         center = L + center     if center < 0 or center >= L:         raise Exception("Invalid value of center in _vector2mps")     err: float = 0.0     for i in range(center):         output[i], ψ, new_err = _left_orth_2site(             ψ.reshape(ψ.shape[0], dimensions[i], -1, ψ.shape[-1]), strategy         )         err += new_err     for i in range(L - 1, center, -1):         ψ, output[i], new_err = _right_orth_2site(             ψ.reshape(ψ.shape[0], -1, dimensions[i], ψ.shape[-1]), strategy         )         err += new_err     if normalize:         N = np.linalg.norm(ψ.reshape(-1))         ψ /= N         err /= float(N)     output[center] = ψ     return output, err   all = ["_destructive_svd", "_schmidt_weights", "_vector2mps"]  and schmidt.pxi:

Edit
I'll convert the schmidt.py file from NumPy to PyTorch. Here's the PyTorch version:


schmidt.py - PyTorch Version
Code 
Here are the key changes I made to convert from NumPy to PyTorch:

Import change: Replaced import numpy as np with import torch
Array conversion:
Changed np.asarray(state).copy().reshape(1, -1, 1) to torch.as_tensor(state).clone().reshape(1, -1, 1)
SVD function:
Replaced _scipy_svd() with torch.linalg.svd() and removed the scipy-specific parameters
PyTorch's SVD returns (U, S, Vh) similar to NumPy/SciPy
Mathematical operations:
Changed np.sum(s) to torch.sum(s)
Changed ψ.size to ψ.numel() (PyTorch equivalent for total number of elements)
Changed np.linalg.norm() to torch.linalg.norm()
Type annotations:
Changed NDArray to torch.Tensor in the type hints (implicitly through the conversion)
The schmidt.pxi file appears to be a Cython extension file with low-level optimizations. Since you mentioned to assume that imported functions are already okay, I'm treating _left_orth_2site and _right_orth_2site as functions that will work with PyTorch tensors when they're converted.

The overall structure and logic remain the same, with only the NumPy-specific operations converted to their PyTorch equivalents.




typing.py: from future import annotations
import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy.sparse as sp  # type: ignore
from typing import TypeAlias, Annotated, TypeVar

Natural: TypeAlias = Annotated[int, ">=1"]

Weight: TypeAlias = float | complex
"""A real or complex number."""

Unitary: TypeAlias = NDArray
"""Unitary matrix in :class:numpy.ndarray dense format."""

SparseOperator: TypeAlias = sp.csr_matrix | sp.bsr_matrix | sp.coo_matrix
"""An operator in sparse matrix format."""

Operator: TypeAlias = NDArray | SparseOperator
"""An operator, either in :class:np.ndarray or sparse matrix format."""

DenseOperator: TypeAlias = NDArray
"""An operator in :class:numpy.ndarray format."""

Vector: TypeAlias = NDArray
"""A one-dimensional :class:numpy.ndarray representing a wavefunction."""

VectorLike: TypeAlias = ArrayLike
"""Any Python type that can be coerced to Vector type."""

Tensor3: TypeAlias = NDArray
""":class:numpy.ndarray tensor with three indices."""

Tensor4: TypeAlias = NDArray
""":class:numpy.ndarray tensor with four indices."""

Environment: TypeAlias = NDArray
"""Left or right environment represented as tensor."""

MPOEnvironment: TypeAlias = NDArray
"""Left or right environment of an MPS-MPO-MPS contraction."""

FloatOrArray = TypeVar("FloatOrArray", float, NDArray[np.floating])

def to_dense_operator(O: Operator) -> DenseOperator:
    if sp.issparse(O):
        return O.toarray()  # type: ignore
    return O

all = [
    "NDArray",
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

Edit
I'll convert the typing.py file from NumPy to PyTorch:


typing.py - PyTorch Version
Code 
Here are the key changes I made to convert from NumPy to PyTorch:

Import changes:
Replaced import numpy as np and from numpy.typing import NDArray, ArrayLike with import torch
Added Union import for better type compatibility
Type alias updates:
Changed all NDArray references to torch.Tensor
Updated VectorLike from ArrayLike to explicit Python types that can be converted to tensors
Updated FloatOrArray TypeVar from NDArray[np.floating] to torch.Tensor
Function conversion:
Updated to_dense_operator() to convert sparse matrices to PyTorch tensors using torch.from_numpy(O.toarray())
Documentation updates:
Updated all docstrings to reference :class:torch.Tensor instead of `:class:`numpy.ndarray
Exports cleanup:
Removed "NDArray" from __all__ since we're no longer using NumPy arrays
The sparse matrix support remains the same since PyTorch doesn't have as comprehensive sparse matrix support as SciPy, so we still use SciPy sparse matrices and convert them to dense PyTorch tensors when needed.







from __future__ import annotations
import torch
import scipy.sparse as sp  # type: ignore
from typing import TypeAlias, Annotated, TypeVar, Union

Natural: TypeAlias = Annotated[int, ">=1"]

Weight: TypeAlias = float | complex
"""A real or complex number."""

Unitary: TypeAlias = torch.Tensor
"""Unitary matrix in :class:`torch.Tensor` dense format."""

SparseOperator: TypeAlias = sp.csr_matrix | sp.bsr_matrix | sp.coo_matrix
"""An operator in sparse matrix format."""

Operator: TypeAlias = torch.Tensor | SparseOperator
"""An operator, either in :class:`torch.Tensor` or sparse matrix format."""

DenseOperator: TypeAlias = torch.Tensor
"""An operator in :class:`torch.Tensor` format."""

Vector: TypeAlias = torch.Tensor
"""A one-dimensional :class:`torch.Tensor` representing a wavefunction."""

VectorLike: TypeAlias = Union[torch.Tensor, list, tuple, float, int, complex]
"""Any Python type that can be coerced to `Vector` type."""

Tensor3: TypeAlias = torch.Tensor
""":class:`torch.Tensor` tensor with three indices."""

Tensor4: TypeAlias = torch.Tensor
""":class:`torch.Tensor` tensor with four indices."""

Environment: TypeAlias = torch.Tensor
"""Left or right environment represented as tensor."""

MPOEnvironment: TypeAlias = torch.Tensor
"""Left or right environment of an MPS-MPO-MPS contraction."""

FloatOrArray = TypeVar("FloatOrArray", float, torch.Tensor)


def to_dense_operator(O: Operator) -> DenseOperator:
    if sp.issparse(O):
        return torch.from_numpy(O.toarray())  # type: ignore
    return O


__all__ = [
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
