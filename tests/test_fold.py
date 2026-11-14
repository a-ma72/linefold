# SPDX-FileCopyrightText: 2026 Andreas Martin
# SPDX-License-Identifier: BSD-2-Clause

"""The ladder, the column scratch, and the stroke that ends on the last sample."""

from __future__ import annotations

import numpy as np

from pyftc.linefold.fold import (
    BASE,
    FINE,
    ColumnCache,
    ColumnScratch,
    _anchored_edges,
    _column_extents,
    _extents_rungs,
    _index_spans,
    envelope,
)
from pyftc.linefold.plot import Plot


def _same(got, raw) -> None:
    assert np.allclose(got[0], raw[0], equal_nan=True)
    assert np.allclose(got[1], raw[1], equal_nan=True)
    assert np.array_equal(got[2], raw[2])


def test_a_wide_view_reads_the_fringe_and_matches_the_scan():
    n = 200_000
    columns = 1_000
    values = np.linspace(-1.0, 1.0, n)
    values[1_000:1_300] = np.nan
    values[40_000:40_050] = 7.0
    values[80_000:80_400] = 2.0
    cache = ColumnCache(values)
    got = envelope(values, 0.0, float(n), columns, cache=cache, anchored=True)
    reads = envelope.samples_read
    filled = envelope.columns_filled
    raw = envelope(values, 0.0, float(n), columns, anchored=True)
    _same(got, raw)
    assert filled == columns
    # A column holds about two hundred samples. The ladder reads the fringe.
    assert reads < n / 10

    shifted = envelope(values, 1.5, float(n) - 10.5, columns, cache=cache, anchored=True)
    fringe = envelope.samples_read
    plain = envelope(values, 1.5, float(n) - 10.5, columns, anchored=True)
    _same(shifted, plain)
    assert 0 < fringe < n / 10


def test_equal_samples_keep_a_count_above_one():
    values = np.ones(8_000)
    values[100:120] = np.nan
    cache = ColumnCache(values)
    got = envelope(values, 3.0, 7_000.0, 40, cache=cache, anchored=True)
    raw = envelope(values, 3.0, 7_000.0, 40, anchored=True)
    _same(got, raw)
    flat = (got[2] > 1) & (got[0] == got[1])
    assert np.any(flat)


def test_a_logarithmic_column_shorter_than_the_base_is_read():
    n = 200_000
    columns = 1_000
    values = np.linspace(-1.0, 2.0, n)
    cache = ColumnCache(values)
    got = envelope(values, 0.0, float(n), columns, cache=cache, log_y=True, anchored=True)
    reads = envelope.samples_read
    raw = envelope(values, 0.0, float(n), columns, log_y=True, anchored=True)
    _same(got, raw)
    # Two hundred samples a column is below BASE, so the fine rungs are not used.
    assert reads > n * 9 // 10


def test_a_wide_logarithmic_column_still_matches():
    n = 20_000
    values = np.linspace(-2.0, 3.0, n)
    values[10] = np.nan
    cache = ColumnCache(values)
    assert n // 8 > BASE
    got = envelope(values, 0.0, float(n), 8, cache=cache, log_y=True, anchored=True)
    raw = envelope(values, 0.0, float(n), 8, log_y=True, anchored=True)
    _same(got, raw)


def test_the_numpy_rungs_match_the_scan():
    values = np.linspace(0.0, 1.0, 50_000)
    values[20:40] = np.nan
    values[-10:] = 3.0
    cache = ColumnCache(values)
    edges = _anchored_edges(2.5, 49_000.5, 80, False)
    starts, ends = _index_spans(edges, values.size)
    mins = np.full(starts.shape[0], np.nan)
    maxs = np.full(starts.shape[0], np.nan)
    counts = np.zeros(starts.shape[0], dtype=np.int64)
    reads = np.zeros(starts.shape[0], dtype=np.int64)
    rungs = list(cache.rungs) + list(cache.fine)
    _extents_rungs(values, starts, ends, mins, maxs, counts, False, rungs, reads)
    raw_mins = np.full(starts.shape[0], np.nan)
    raw_maxs = np.full(starts.shape[0], np.nan)
    raw_counts = np.zeros(starts.shape[0], dtype=np.int64)
    _column_extents(values, starts, ends, raw_mins, raw_maxs, raw_counts, False)
    assert np.allclose(mins, raw_mins, equal_nan=True)
    assert np.allclose(maxs, raw_maxs, equal_nan=True)
    assert np.array_equal(counts, raw_counts)
    assert int(reads.sum()) < values.size / 10
    assert cache.fine[-1].nominal == FINE


def test_a_whole_column_pan_fills_only_the_new_strip():
    n = 200_000
    columns = 1_000
    values = np.linspace(0.0, 1.0, n)
    cache = ColumnCache(values)
    scratch = ColumnScratch()
    width = float(n) / columns
    envelope(values, 0.0, float(n), columns, cache=cache, anchored=True, scratch=scratch)
    assert envelope.columns_filled == columns

    envelope(values, 0.0, float(n), columns, cache=cache, anchored=True, scratch=scratch)
    assert envelope.samples_read == 0
    assert envelope.columns_filled == 0

    stayed = envelope(
        values, 0.4 * width, float(n) + 0.4 * width, columns,
        cache=cache, anchored=True, scratch=scratch,
    )
    assert envelope.columns_filled == 0
    assert envelope.samples_read == 0
    plain = envelope(values, 0.4 * width, float(n) + 0.4 * width, columns, cache=cache, anchored=True)
    _same(stayed, plain)

    moved = envelope(
        values, width, float(n) + width, columns, cache=cache, anchored=True, scratch=scratch,
    )
    assert envelope.columns_filled == 1
    assert envelope.samples_read < FINE * 2
    plain = envelope(values, width, float(n) + width, columns, cache=cache, anchored=True)
    _same(moved, plain)

    envelope(values, 0.0, float(n) / 2.0, columns, cache=cache, anchored=True, scratch=scratch)
    assert envelope.columns_filled == columns

    envelope(values, 0.0, float(n), columns, cache=cache, anchored=True, scratch=scratch, log_y=True)
    assert envelope.columns_filled == columns
    assert envelope.samples_read > n * 9 // 10


def _finite_x(plot: Plot) -> np.ndarray:
    xs = plot.data_points()[:, 0]
    return xs[np.isfinite(xs)]


def test_the_stroke_ends_on_the_last_sample():
    values = np.ones(50)
    plot = Plot()
    plot.add_line(values)
    plot.set_pane(10, 100, 1.0)
    plot.set_range(0.0, 200.0, -1.0, 2.0)
    plot.project()
    finite = _finite_x(plot)
    assert finite.size
    assert finite.max() == 49.0
    assert finite.max() < plot.view_max_x()


def test_the_stroke_ends_on_the_last_time():
    values = np.ones(50)
    times = np.arange(50, dtype=np.float64) * 2.0
    plot = Plot()
    plot.add_line(values)
    plot.set_x(times)
    plot.set_pane(10, 80, 1.0)
    plot.set_range(0.0, 400.0, -1.0, 2.0)
    plot.project()
    finite = _finite_x(plot)
    assert finite.size
    assert finite.max() == 98.0
    assert finite.max() < plot.view_max_x()
