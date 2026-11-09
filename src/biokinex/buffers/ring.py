"""Fixed-capacity circular buffer with zero-copy access to recent samples.

The buffer allocates ``2 * capacity`` rows once and writes every sample twice: at index ``i`` and
at ``i + capacity``. Because of this mirror, the newest ``k`` samples (for any ``k <= capacity``)
always sit in one contiguous block, so they can be returned as a NumPy view without copying or
joining two pieces. Memory use never changes after construction.

Views returned by :meth:`RingBuffer.latest` are read-only and point into the live storage. They
stay valid only until the next write; copy them if you need to keep the values longer.

The buffer is not thread-safe. In BioKineX each pipeline owns its buffers and uses them from a
single event loop.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from biokinex.core.types import FloatArray

__all__ = ["RingBuffer"]

_MIRROR_COPIES: Final = 2


class RingBuffer:
    """Circular buffer of ``float64`` rows with a fixed capacity.

    Args:
        capacity: Largest number of rows kept; older rows are overwritten.
        row_shape: Shape of one row, for example ``()`` for scalars or ``(3,)`` for vectors.
    """

    __slots__ = ("_capacity", "_row_shape", "_size", "_storage", "_total_written", "_write_index")

    def __init__(self, capacity: int, row_shape: tuple[int, ...] = ()) -> None:
        if capacity < 1:
            msg = f"capacity must be >= 1, got {capacity}"
            raise ValueError(msg)
        if any(dim < 1 for dim in row_shape):
            msg = f"row_shape dimensions must be >= 1, got {row_shape}"
            raise ValueError(msg)
        self._capacity = capacity
        self._row_shape = tuple(row_shape)
        self._storage: FloatArray = np.zeros((_MIRROR_COPIES * capacity, *row_shape))
        self._write_index = 0
        self._size = 0
        self._total_written = 0

    @property
    def capacity(self) -> int:
        """Largest number of rows the buffer keeps."""
        return self._capacity

    @property
    def row_shape(self) -> tuple[int, ...]:
        """Shape of one row."""
        return self._row_shape

    @property
    def is_full(self) -> bool:
        """Whether the next write will overwrite the oldest row."""
        return self._size == self._capacity

    @property
    def total_written(self) -> int:
        """Number of rows written since construction or the last :meth:`clear`."""
        return self._total_written

    @property
    def nbytes(self) -> int:
        """Size of the preallocated storage in bytes; it never changes."""
        return int(self._storage.nbytes)

    def __len__(self) -> int:
        return self._size

    def append(self, row: npt.ArrayLike) -> None:
        """Write one row, overwriting the oldest row when the buffer is full."""
        values = self._as_rows(row, self._row_shape, "row")
        index = self._write_index
        self._storage[index] = values
        self._storage[index + self._capacity] = values
        self._advance(1)

    def extend(self, rows: npt.ArrayLike) -> None:
        """Write several rows in order, oldest first.

        If more than ``capacity`` rows are given, only the newest ``capacity`` rows are kept,
        which gives the same result as appending them one by one.
        """
        values = self._as_rows(rows, (-1, *self._row_shape), "rows")
        count = values.shape[0]
        if count == 0:
            return
        skipped = max(0, count - self._capacity)
        # Rows that would be overwritten before anyone could read them are not copied at all.
        values = values[skipped:]
        kept = values.shape[0]

        start = self._write_index
        first = min(kept, self._capacity - start)
        rest = kept - first
        # Each block goes into both halves of the storage to keep the mirror complete.
        self._storage[start : start + first] = values[:first]
        self._storage[start + self._capacity : start + self._capacity + first] = values[:first]
        self._storage[:rest] = values[first:]
        self._storage[self._capacity : self._capacity + rest] = values[first:]

        self._total_written += skipped
        self._advance(kept)

    def latest(self, count: int | None = None) -> FloatArray:
        """Return the newest ``count`` rows, oldest first, as a read-only zero-copy view.

        ``None`` returns every stored row.

        Raises:
            ValueError: If ``count`` is negative or larger than the number of stored rows.
        """
        if count is None:
            count = self._size
        elif not 0 <= count <= self._size:
            msg = f"count must be in [0, {self._size}], got {count}"
            raise ValueError(msg)
        # The newest row is just before the write index in the upper half, so the block never
        # crosses the end of the storage.
        end = self._write_index + self._capacity
        view = self._storage[end - count : end]
        view.flags.writeable = False
        return view

    def last(self) -> FloatArray:
        """Return the newest row as a read-only view.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            msg = "last() called on an empty buffer"
            raise IndexError(msg)
        # Plain indexing would turn a scalar row into a copied NumPy scalar; reshape keeps a view.
        return self.latest(1).reshape(self._row_shape)

    def clear(self) -> None:
        """Remove every row. The storage is kept, so no memory is allocated."""
        self._write_index = 0
        self._size = 0
        self._total_written = 0

    def _advance(self, count: int) -> None:
        """Move the write position forward after ``count`` rows were stored."""
        self._write_index = (self._write_index + count) % self._capacity
        self._size = min(self._size + count, self._capacity)
        self._total_written += count

    @staticmethod
    def _as_rows(data: npt.ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
        """Convert ``data`` to float64 and check its shape.

        NumPy would broadcast a scalar into a whole row, which hides caller bugs, so the shape
        must match exactly. A ``-1`` in ``shape`` accepts any length on that axis.
        """
        values = np.asarray(data, dtype=np.float64)
        # An exact match is the common case on the hot path, so it is checked first.
        if values.shape == shape:
            return values
        matches = values.ndim == len(shape) and all(
            expected in (-1, actual) for expected, actual in zip(shape, values.shape, strict=True)
        )
        if not matches:
            expected_text = tuple("N" if dim == -1 else dim for dim in shape)
            msg = f"{name} must have shape {expected_text}, got {values.shape}"
            raise ValueError(msg)
        return values
