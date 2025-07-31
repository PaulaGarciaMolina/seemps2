from __future__ import annotations
import torch
from collections.abc import Sequence, Iterable, Iterator
from typing import overload, TypeVar

_T = TypeVar("_T", bound="TensorArray")


class TensorArray(Sequence[torch.Tensor]):
    """TensorArray class.

    This class provides the basis for all tensor networks. The class abstracts
    a one-dimensional array of tensors that is freshly copied whenever the
    object is cloned. Two TensorArray's can share the same tensors and be
    destructively modified.

    Parameters
    ----------
    data: Iterable[torch.Tensor]
        Any sequence of tensors that can be stored in this object. They are
        not checked to have the right number of dimensions. This sequence is
        cloned to avoid nasty side effects when destructively modifying it.
    device: torch.device, optional
        Device to store tensors on. If None, tensors keep their original device.
    dtype: torch.dtype, optional
        Data type for tensors. If None, tensors keep their original dtype.
    """

    _data: list[torch.Tensor]
    size: int
    device: torch.device | None
    dtype: torch.dtype | None

    def __init__(
        self, 
        data: Iterable[torch.Tensor], 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        self._data = []
        self.device = device
        self.dtype = dtype
        
        for tensor in data:
            # Convert to specified device/dtype if provided
            if device is not None:
                tensor = tensor.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            self._data.append(tensor)
        
        self.size = len(self._data)

    @overload
    def __getitem__(self, k: int) -> torch.Tensor: ...

    @overload
    def __getitem__(self, k: slice) -> Sequence[torch.Tensor]: ...

    def __getitem__(self, k: int | slice) -> torch.Tensor | Sequence[torch.Tensor]:
        #
        # Get tensor at position `k`. If 'A' is a TensorArray, we can now
        # do A[k]
        #
        return self._data[k]  # type: ignore

    def __setitem__(self, k: int, value: torch.Tensor) -> torch.Tensor:
        #
        # Replace tensor at position `k` with new tensor `value`. If 'A'
        # is a TensorArray, we can now do A[k] = value
        #
        # Ensure the new tensor is on the same device/dtype if specified
        if self.device is not None:
            value = value.to(self.device)
        if self.dtype is not None:
            value = value.to(self.dtype)
        
        self._data[k] = value
        return value

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self._data.__iter__()

    def __reversed__(self) -> Iterator[torch.Tensor]:
        return self._data.__reversed__()

    def __len__(self) -> int:
        return self.size

    def to(self, device: torch.device | str) -> TensorArray:
        """Move all tensors to the specified device."""
        return TensorArray([tensor.to(device) for tensor in self._data], device=torch.device(device))

    def cuda(self) -> TensorArray:
        """Move all tensors to CUDA device."""
        return self.to('cuda')

    def cpu(self) -> TensorArray:
        """Move all tensors to CPU."""
        return self.to('cpu')

    def detach(self) -> TensorArray:
        """Detach all tensors from the computation graph."""
        return TensorArray([tensor.detach() for tensor in self._data], device=self.device, dtype=self.dtype)