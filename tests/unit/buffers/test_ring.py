"""Tests for biokinex.buffers.ring."""

from __future__ import annotations

import tracemalloc
from collections import deque

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from biokinex.buffers.ring import RingBuffer


def filled(
    capacity: int, values: range | list[float], row_shape: tuple[int, ...] = ()
) -> RingBuffer:
    """Return a buffer after appending ``values`` one by one."""
    buffer = RingBuffer(capacity, row_shape)
    for value in values:
        buffer.append(value)
    return buffer


class TestConstruction:
    def test_starts_empty(self) -> None:
        buffer = RingBuffer(4, (3,))
        assert len(buffer) == 0
        assert buffer.capacity == 4
        assert buffer.row_shape == (3,)
        assert not buffer.is_full
        assert buffer.total_written == 0
        assert buffer.latest().shape == (0, 3)

    def test_preallocates_mirrored_storage(self) -> None:
        assert RingBuffer(100, (3,)).nbytes == 2 * 100 * 3 * 8

    @pytest.mark.parametrize("capacity", [0, -1])
    def test_rejects_invalid_capacity(self, capacity: int) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            RingBuffer(capacity)

    @pytest.mark.parametrize("row_shape", [(0,), (3, 0), (-1,)])
    def test_rejects_invalid_row_shape(self, row_shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="row_shape dimensions"):
            RingBuffer(4, row_shape)


class TestAppend:
    def test_keeps_rows_oldest_first_before_full(self) -> None:
        buffer = filled(4, [1.0, 2.0, 3.0])
        assert_array_equal(buffer.latest(), [1.0, 2.0, 3.0])
        assert not buffer.is_full

    def test_overwrites_oldest_after_wrap(self) -> None:
        buffer = filled(3, range(7))
        assert_array_equal(buffer.latest(), [4.0, 5.0, 6.0])
        assert buffer.is_full
        assert len(buffer) == 3
        assert buffer.total_written == 7

    def test_capacity_one(self) -> None:
        buffer = filled(1, [5.0, 6.0])
        assert_array_equal(buffer.latest(), [6.0])

    def test_vector_rows(self) -> None:
        buffer = filled(2, [], row_shape=(3,))
        buffer.append([1, 2, 3])
        buffer.append(np.array([4.0, 5.0, 6.0]))
        buffer.append([7, 8, 9])
        assert_array_equal(buffer.latest(), [[4, 5, 6], [7, 8, 9]])
        assert buffer.latest().dtype == np.float64

    @pytest.mark.parametrize("row", [1.0, [1.0, 2.0], [[1.0, 2.0, 3.0]]])
    def test_rejects_wrong_row_shape(self, row: object) -> None:
        buffer = RingBuffer(4, (3,))
        with pytest.raises(ValueError, match=r"row must have shape \(3,\)"):
            buffer.append(row)
        assert len(buffer) == 0

    def test_last_scalar_row_is_a_view(self) -> None:
        buffer = filled(3, range(5))
        newest = buffer.last()
        assert newest.shape == ()
        assert float(newest) == 4.0
        assert np.shares_memory(newest, buffer.latest())

    def test_last_vector_row(self) -> None:
        buffer = filled(3, [], row_shape=(3,))
        buffer.extend([[1, 2, 3], [4, 5, 6]])
        assert_array_equal(buffer.last(), [4.0, 5.0, 6.0])

    def test_last_on_empty_buffer_raises(self) -> None:
        with pytest.raises(IndexError, match="empty buffer"):
            RingBuffer(3).last()


class TestExtend:
    def test_fits_without_wrap(self) -> None:
        buffer = RingBuffer(5)
        buffer.extend([1.0, 2.0, 3.0])
        assert_array_equal(buffer.latest(), [1.0, 2.0, 3.0])

    def test_wraps_around_the_end(self) -> None:
        buffer = filled(4, [0.0, 1.0, 2.0])
        buffer.extend([3.0, 4.0, 5.0])
        assert_array_equal(buffer.latest(), [2.0, 3.0, 4.0, 5.0])

    def test_more_rows_than_capacity_keeps_newest(self) -> None:
        buffer = filled(3, [100.0])
        buffer.extend(np.arange(10.0))
        assert_array_equal(buffer.latest(), [7.0, 8.0, 9.0])
        assert buffer.total_written == 11

    def test_empty_extend_changes_nothing(self) -> None:
        buffer = filled(3, [1.0])
        buffer.extend(np.zeros(0))
        assert_array_equal(buffer.latest(), [1.0])
        assert buffer.total_written == 1

    def test_vector_rows(self) -> None:
        buffer = RingBuffer(3, (2,))
        buffer.extend([[1, 1], [2, 2], [3, 3], [4, 4]])
        assert_array_equal(buffer.latest(), [[2, 2], [3, 3], [4, 4]])

    @pytest.mark.parametrize("rows", [[1.0, 2.0], [[1.0, 2.0]], 5.0])
    def test_rejects_wrong_shape(self, rows: object) -> None:
        buffer = RingBuffer(4, (3,))
        with pytest.raises(ValueError, match=r"rows must have shape \('N', 3\)"):
            buffer.extend(rows)


class TestLatest:
    def test_count_selects_newest_rows(self) -> None:
        buffer = filled(4, range(6))
        assert_array_equal(buffer.latest(2), [4.0, 5.0])
        assert_array_equal(buffer.latest(0), np.zeros(0))

    @pytest.mark.parametrize("count", [-1, 4])
    def test_rejects_invalid_count(self, count: int) -> None:
        buffer = filled(5, range(3))
        with pytest.raises(ValueError, match=r"count must be in \[0, 3\]"):
            buffer.latest(count)

    def test_is_contiguous_after_wrap(self) -> None:
        buffer = filled(4, range(6), row_shape=())
        view = buffer.latest()
        assert view.flags.c_contiguous
        assert_array_equal(view, [2.0, 3.0, 4.0, 5.0])

    def test_returns_views_into_shared_storage(self) -> None:
        buffer = filled(4, range(6))
        assert np.shares_memory(buffer.latest(), buffer.latest(2))

    def test_view_is_read_only(self) -> None:
        buffer = filled(4, range(3))
        with pytest.raises(ValueError, match="read-only"):
            buffer.latest()[0] = 99.0
        with pytest.raises(ValueError, match="read-only"):
            buffer.last()[...] = 99.0

    def test_view_reflects_later_writes(self) -> None:
        """Documents the zero-copy contract: views are only valid until the next write."""
        buffer = filled(2, [1.0, 2.0])
        view = buffer.latest()
        snapshot = view.copy()
        buffer.extend([3.0, 4.0])
        assert_array_equal(snapshot, [1.0, 2.0])
        assert not np.array_equal(view, snapshot)

    def test_buffer_stays_writable_after_handing_out_views(self) -> None:
        buffer = filled(2, [1.0])
        buffer.latest()
        buffer.append(2.0)
        assert_array_equal(buffer.latest(), [1.0, 2.0])


def test_clear_resets_state_but_keeps_storage() -> None:
    buffer = filled(3, range(5))
    nbytes = buffer.nbytes
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.total_written == 0
    assert buffer.nbytes == nbytes
    buffer.append(9.0)
    assert_array_equal(buffer.latest(), [9.0])


def test_sustained_streaming_does_not_allocate() -> None:
    buffer = RingBuffer(512, (3,))
    row = np.array([0.1, 0.2, 0.3])
    block = np.ones((10, 3))
    for _ in range(1_000):
        buffer.append(row)

    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(50_000):
            buffer.append(row)
            buffer.extend(block)
            buffer.latest(64)
        growth = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    # A small allowance covers interpreter caches; a leak would grow by megabytes.
    assert growth < 16 * 1024


operations = st.lists(
    st.one_of(
        st.tuples(st.just("append"), st.integers(-100, 100)),
        st.tuples(st.just("extend"), st.lists(st.integers(-100, 100), max_size=20)),
        st.tuples(st.just("clear"), st.none()),
    ),
    max_size=40,
)


@given(st.integers(1, 8), operations, st.data())
def test_matches_bounded_deque_model(
    capacity: int,
    ops: list[tuple[str, int | list[int] | None]],
    data: st.DataObject,
) -> None:
    buffer = RingBuffer(capacity)
    model: deque[float] = deque(maxlen=capacity)
    for name, argument in ops:
        if name == "append":
            assert isinstance(argument, int)
            buffer.append(argument)
            model.append(argument)
        elif name == "extend":
            assert isinstance(argument, list)
            buffer.extend(np.array(argument, dtype=np.float64))
            model.extend(argument)
        else:
            buffer.clear()
            model.clear()
        assert len(buffer) == len(model)
        assert buffer.is_full == (len(model) == capacity)
        assert_array_equal(buffer.latest(), list(model))
        count = data.draw(st.integers(0, len(model)))
        assert_array_equal(buffer.latest(count), list(model)[len(model) - count :])
