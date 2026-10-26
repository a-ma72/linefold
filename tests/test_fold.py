# SPDX-FileCopyrightText: 2026 Andreas Martin
# SPDX-License-Identifier: BSD-2-Clause

"""Checks for the column envelope and the camera that sits on it."""

from __future__ import annotations

import math

import numpy as np

from linefold.fold import (
    BASE,
    FAN_FOUR,
    FAN_TAPER,
    FAN_TWO,
    PATTERN,
    ColumnCache,
    _column_extents,
    _column_extents_numpy,
    _edges,
    _extents_buckets,
    _index_spans,
    _limits_numpy,
    _monotonic_numpy,
    _rungs,
    chosen_pattern,
    _anchored_edges,
    envelope,
    limits,
    monotonic,
    schedule_cost,
)
from linefold.mpl import _ELogFormatter, _Y_LABEL, _Y_TICK, _held
from linefold.plot import Plot


def _counts(data, columns):
    _mins, _maxs, counts, _edges = envelope(data, 0.0, float(data.size), columns)
    return counts


def test_fair_split():
    data = np.zeros(100, dtype=np.float64)
    counts = _counts(data, 7)
    assert counts.sum() == 100
    assert counts.max() - counts.min() <= 1


def test_chord_is_not_half():
    # Five samples, three columns. The edge at 10/3 sits a third of the way
    # from the 0 to the 10, so the column grows by 10/3 and not by 5.
    data = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    mins, maxs, _counts, edges = envelope(data, 0.0, 5.0, 3)
    assert math.isclose(edges[2], 10.0 / 3.0)
    assert math.isclose(maxs[1], 10.0 / 3.0)
    assert maxs[1] < 5.0


def test_nan_is_not_an_extreme():
    data = np.array([1.0, math.nan, 3.0])
    mins, maxs, counts, _edges = envelope(data, 0.0, 3.0, 1)
    assert counts[0] == 2
    assert math.isclose(mins[0], 1.0)
    assert math.isclose(maxs[0], 3.0)


def test_time_base_bisects():
    data = np.arange(10, dtype=np.float64)
    time = data * 2.0
    mins, _maxs, counts, _edges = envelope(data, 5.0, 15.0, 2, x=time)
    # x = 6, 8 in the first column; 10, 12, 14 in the second.
    # The edge at 5 lies halfway from x=4 to x=6, so the column reaches 2.5.
    assert counts[0] == 2
    assert counts[1] == 3
    assert math.isclose(mins[0], 2.5)
    assert math.isclose(mins[1], 5.0)


def test_log_edges_are_geometric():
    data = np.linspace(1.0, 1000.0, 1000)
    _mins, _maxs, _counts, edges = envelope(data, 1.0, 1000.0, 4, log_x=True)
    ratios = edges[1:] / edges[:-1]
    assert np.allclose(ratios, ratios[0])


def test_log_y_drops_non_positive():
    data = np.array([-2.0, 0.0, 4.0, 8.0])
    mins, maxs, counts, _edges = envelope(data, 0.0, 4.0, 1, log_y=True)
    assert counts[0] == 2
    assert math.isclose(mins[0], 4.0)
    assert math.isclose(maxs[0], 8.0)


def _scan(fn, data, starts, ends, positive_only):
    mins = np.full(starts.size, np.nan)
    maxs = np.full(starts.size, np.nan)
    counts = np.zeros(starts.size, dtype=np.int64)
    fn(data, starts, ends, mins, maxs, counts, positive_only)
    return mins, maxs, counts


def test_numpy_scan_matches_the_one_in_use():
    # The NumPy path is what runs when numba is missing. It has to name the
    # same extremes, including a NaN, an empty column and a logarithmic one.
    data = np.array([1.0, np.nan, -2.0, 4.0, np.inf, 0.0, 3.0])
    starts = np.array([0, 3, 5, 5], dtype=np.int64)
    ends = np.array([3, 5, 5, 7], dtype=np.int64)
    for positive in (False, True):
        got = _scan(_column_extents_numpy, data, starts, ends, positive)
        used = _scan(_column_extents, data, starts, ends, positive)
        for left, right in zip(got, used):
            assert np.allclose(left, right, equal_nan=True)
    # A gap between columns is not a partition, so the slices are gathered.
    gap_starts = np.array([0, 4], dtype=np.int64)
    gap_ends = np.array([2, 7], dtype=np.int64)
    got = _scan(_column_extents_numpy, data, gap_starts, gap_ends, False)
    used = _scan(_column_extents, data, gap_starts, gap_ends, False)
    for left, right in zip(got, used):
        assert np.allclose(left, right, equal_nan=True)
    assert _monotonic_numpy(np.array([0.0, 1.0, 1.0])) == bool(monotonic(np.array([0.0, 1.0, 1.0])))
    assert _monotonic_numpy(np.array([0.0, np.nan, 2.0])) == bool(monotonic(np.array([0.0, np.nan, 2.0])))
    assert not _monotonic_numpy(np.array([0.0, 2.0, 1.0]))
    left = _limits_numpy(data)
    right = limits(data)
    assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=0.0) or (math.isinf(a) and math.isinf(b) and a == b)
               for a, b in zip(left, right))


def _same_envelope(data, x0, x1, columns, cache, x=None, log_y=False):
    plain = envelope(data, x0, x1, columns, x=x, log_y=log_y)
    cached = envelope(data, x0, x1, columns, x=x, log_y=log_y, cache=cache)
    for left, right in zip(plain, cached):
        assert np.allclose(left, right, equal_nan=True)


def test_cached_envelope_matches_the_scan():
    # The ladder answers a column. The chord at its edges is still the two
    # raw samples, so the fold is the one the scan draws, including a window
    # that starts in the middle of a bucket.
    rng = np.random.default_rng(7)
    data = rng.normal(size=20000)
    data[::80] = np.nan
    data[::200] = -3.0
    windows = (
        (0.0, 20000.0, 3),
        (0.0, 20000.0, 100),
        (100.0, 18000.0, 7),
        (500.5, 15000.25, 4),
    )
    time = np.arange(data.size, dtype=np.float64) * 0.5
    for pattern in (FAN_FOUR, FAN_TWO, FAN_TAPER):
        cache = ColumnCache(data, pattern)
        geometry = _rungs(data.size, pattern)
        assert len(geometry) == len(cache.rungs)
        for rung_geometry, rung in zip(geometry, cache.rungs):
            nominal, starts, ends = rung_geometry
            assert nominal == rung.nominal
            assert np.array_equal(starts, rung.starts)
            assert np.array_equal(ends, rung.ends)
        base = cache.rungs[-1]
        assert np.allclose(base.entry[0], data[0], equal_nan=True)
        assert np.allclose(base.exit[0], data[BASE - 1], equal_nan=True)
        for x0, x1, columns in windows:
            for log_y in (False, True):
                _same_envelope(data, x0, x1, columns, cache, log_y=log_y)
        _same_envelope(data, 100.0, 4000.0, 5, cache, x=time)


def test_the_numpy_ladder_matches_the_scan():
    # Without numba the buckets are a reduction, not a pass over the samples.
    # The short last bucket, a column that starts inside one, and a logarithmic
    # axis have to name the same extremes as the scan.
    rng = np.random.default_rng(11)
    data = rng.normal(size=BASE * 5 + 100)
    data[::40] = np.nan
    data[::90] = -4.0
    cache = ColumnCache(data, FAN_TAPER)
    rung = cache.rungs[-1]
    windows = (
        (0.0, float(data.size), 3),
        (0.0, float(data.size), 9),
        (50.0, float(data.size) - 20.0, 6),
        (float(BASE) - 10.0, float(data.size), 4),
    )
    for x0, x1, columns in windows:
        edges = _edges(x0, x1, columns, False)
        starts, ends = _index_spans(edges, data.size)
        for positive in (False, True):
            got = _scan(_extents_buckets_of(rung), data, starts, ends, positive)
            used = _scan(_column_extents, data, starts, ends, positive)
            for left, right in zip(got, used):
                assert np.allclose(left, right, equal_nan=True)
    # Overlapping columns, an empty one, and a column that ends inside the short tail.
    starts = np.array([0, 10, BASE + 3, data.size - 80, 4], dtype=np.int64)
    ends = np.array([data.size, BASE * 3 + 7, BASE * 4, data.size, 4], dtype=np.int64)
    for positive in (False, True):
        got = _scan(_extents_buckets_of(rung), data, starts, ends, positive)
        used = _scan(_column_extents, data, starts, ends, positive)
        for left, right in zip(got, used):
            assert np.allclose(left, right, equal_nan=True)


def _extents_buckets_of(rung):
    def run(data, starts, ends, mins, maxs, counts, positive_only):
        _extents_buckets(data, starts, ends, mins, maxs, counts, positive_only, rung)
    return run


def test_panning_keeps_the_column():
    # A shift smaller than a column leaves every sample where it was. A shift
    # of one column moves the line along by one column, it does not stir it.
    rng = np.random.default_rng(4)
    data = rng.normal(size=20000)
    data[::50] = np.nan
    columns = 50
    span = 10000.0
    x0 = 1000.0
    held = envelope(data, x0, x0 + span, columns, anchored=True)
    nudged = envelope(data, x0 + 40.0, x0 + span + 40.0, columns, anchored=True)
    assert np.allclose(held[0], nudged[0], equal_nan=True)
    assert np.allclose(held[1], nudged[1], equal_nan=True)
    assert np.array_equal(held[2], nudged[2])
    stepped = envelope(data, x0 + 200.0, x0 + span + 200.0, columns, anchored=True)
    assert np.allclose(held[0][1:], stepped[0][:-1], equal_nan=True)
    assert np.allclose(held[1][1:], stepped[1][:-1], equal_nan=True)
    edges = _anchored_edges(x0, x0 + span, columns, False)
    assert edges is not None and len(edges) == columns + 1
    assert np.allclose(np.diff(edges), span / columns)


def test_the_ladder_keeps_the_smaller_schedule():
    # Stored buckets and buckets read, over the same spans. The taper is kept
    # when it reads no more than a factor of two and stores no more than a
    # factor of four. Otherwise the factor of four stays: the small rungs are
    # where the buckets are.
    stored, reads = schedule_cost()
    chosen = chosen_pattern()
    assert PATTERN == chosen
    assert stored["four"] <= stored["two"]
    assert all(value > 0 for value in stored.values())
    assert all(value > 0 for value in reads.values())
    if reads["taper"] <= reads["two"] and stored["taper"] <= stored["four"]:
        assert chosen == FAN_TAPER
    else:
        assert chosen == FAN_FOUR
    line = np.zeros(BASE, dtype=np.float64)
    assert ColumnCache(line).pattern == chosen


def test_a_wide_column_uses_the_ladder():
    plot = Plot()
    rng = np.random.default_rng(1)
    plot.add_line(rng.normal(size=8000))
    plot.set_pane(3, 100, 1.0)
    plot.reset_view()
    _pixels, runs, _ticks = plot.project()
    assert runs[0].summarised
    cache = plot._lines[0].column_cache()
    assert cache is not None
    assert cache.pattern == chosen_pattern()
    first = plot.data_points().copy()
    plot.project()
    assert np.allclose(first, plot.data_points(), equal_nan=True)


def test_wheel_keeps_the_value_under_the_pointer():
    plot = Plot()
    plot.add_line(np.linspace(-1.0, 1.0, 1000))
    plot.set_pane(800, 400, 1.0)
    plot.reset_view()
    before = plot.data_x_at(200)
    span = plot.view_max_x() - plot.view_min_x()
    plot.wheel(200, 100, 2.0, True, False, False)
    after = plot.data_x_at(200)
    assert math.isclose(before, after, rel_tol=1e-9, abs_tol=1e-9)
    assert plot.view_max_x() - plot.view_min_x() < span * 0.6


def test_pan_and_reset_and_rect():
    plot = Plot()
    plot.add_line(np.linspace(0.0, 10.0, 500))
    plot.set_pane(800, 400, 1.0)
    plot.reset_view()
    full = (plot.view_min_x(), plot.view_max_x(), plot.view_min_y(), plot.view_max_y())
    plot.pan(40, 0)
    assert plot.view_min_x() < full[0]
    plot.zoom_rect(100, 50, 300, 250)
    assert plot.view_max_x() - plot.view_min_x() < full[1] - full[0]
    assert not plot.zoom_rect(10, 10, 15, 12)
    plot.reset_view()
    back = (plot.view_min_x(), plot.view_max_x(), plot.view_min_y(), plot.view_max_y())
    assert all(math.isclose(a, b) for a, b in zip(full, back))


def test_project_vertical_and_time():
    plot = Plot()
    y = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    index = plot.add_line(y, colour=(1, 2, 3), name="a")
    plot.set_line_x(index, np.arange(5, dtype=np.float64))
    plot.set_pane(3, 100, 1.0)
    plot.reset_view()
    _pixels, runs, ticks = plot.project()
    axes = {tick.axis for tick in ticks}
    assert 0 in axes and 1 in axes
    assert len(runs) == 1
    data = plot.data_points()
    # A column that holds several samples is still a vertical pair.
    finite = data[np.isfinite(data[:, 1])]
    assert finite.shape[0] >= 2
    assert np.any(np.isclose(finite[:-1, 0], finite[1:, 0]))
    assert plot.run_lines()[0] == 0
    try:
        plot.set_line_x(index, np.array([0.0, 1.0]))
        raise AssertionError("a short time base was accepted")
    except ValueError:
        pass


def test_close_zoom_joins_the_samples():
    plot = Plot()
    y = np.array([0.0, 1.0, np.nan, 4.0, 5.0])
    plot.add_line(y)
    plot.set_pane(400, 100, 1.0)
    plot.reset_view()
    _pixels, runs, _ticks = plot.project()
    assert len(runs) == 1
    assert not runs[0].summarised
    data = plot.data_points()
    finite = data[np.isfinite(data[:, 0])]
    assert np.allclose(finite[:, 0], [0.0, 1.0, 3.0, 4.0])
    assert np.allclose(finite[:, 1], [0.0, 1.0, 4.0, 5.0])
    split = np.where(~np.isfinite(data[:, 0]))[0]
    assert split.size == 1
    assert split[0] == 2


def _has_joined_run(data) -> bool:
    run = []
    for x, _y in data:
        if not np.isfinite(x):
            if len(run) >= 3 and max(run) > min(run):
                return True
            run = []
            continue
        run.append(float(x))
    return len(run) >= 3 and max(run) > min(run)


def test_log_axis_joins_sparse_columns():
    # A logarithmic x crowds samples into the left columns and leaves the
    # right ones with one sample or none. Those empty columns stay on the chord.
    plot = Plot()
    plot.add_line(np.linspace(1.0, 100.0, 500))
    plot.set_pane(400, 200, 1.0)
    plot.set_x_log(True)
    plot.reset_view()
    _pixels, runs, _ticks = plot.project()
    assert runs[0].summarised
    assert _has_joined_run(plot.data_points())

    plot = Plot()
    plot.add_line(np.linspace(1.0, 50.0, 40))
    plot.set_pane(400, 200, 1.0)
    plot.set_y_log(True)
    plot.reset_view()
    plot.set_range(x0=10.0, x1=14.0)
    _pixels, runs, _ticks = plot.project()
    assert not runs[0].summarised
    data = plot.data_points()
    finite = data[np.isfinite(data[:, 0])]
    assert finite.shape[0] >= 2
    assert np.all(np.diff(finite[:, 0]) > 0.0)


def test_flat_log_front_stays_a_line():
    # The front of a logarithmic time axis is the narrow columns. A small
    # amplitude is shorter than the pen there, so the envelope would be a dot.
    count = 100000
    time = np.linspace(0.01, 20.0, count)
    y = 1.0 + 0.01 * np.sin(np.linspace(0.0, 400.0, count))
    plot = Plot()
    index = plot.add_line(y)
    plot.set_line_x(index, time)
    plot.set_pane(500, 250, 1.0)
    plot.set_x_log(True)
    plot.reset_view()
    plot.set_range(x0=0.01, x1=0.5)
    _pixels, runs, _ticks = plot.project()
    assert runs[0].summarised
    assert _has_joined_run(plot.data_points())


def test_shift_claims_an_axis_and_puts_it_back():
    plot = Plot()
    index = plot.add_line(np.linspace(0.0, 10.0, 500))
    plot.set_pane(800, 400, 1.0)
    plot.reset_view()
    plot.shift_line(index, 0.0, 40.0)
    assert plot.own_axis(index)
    line = plot._lines[index]
    assert line.shift_y != 0.0
    _pixels, _runs, ticks = plot.project()
    assert any(tick.axis == 2 and tick.series == index for tick in ticks)
    line.scale_y = 2.0
    line.y_log = True
    plot.set_own_axis(index, False)
    assert not plot.own_axis(index)
    assert line.shift_y == 0.0
    assert line.scale_y == 1.0
    assert line.y_log is False


def test_shared_time_leaves_a_private_axis():
    plot = Plot()
    shared_line = plot.add_line(np.arange(4, dtype=np.float64))
    private = plot.add_line(np.arange(4, dtype=np.float64))
    own = np.arange(4, dtype=np.float64) * 2
    plot.set_line_x(private, own)
    shared = np.arange(4, dtype=np.float64) + 10
    plot.set_x(shared)
    assert np.allclose(plot._lines[shared_line].x, shared)
    assert np.allclose(plot._lines[private].x, own)
    plot.set_x(None)
    assert plot._lines[shared_line].x is None
    assert np.allclose(plot._lines[private].x, own)


def test_y_axis_text_is_smaller_than_the_x_axis():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from linefold.mpl import MplView

    figure = Figure()
    view = MplView(figure=figure)
    view.plot.add_line([1.0, 2.0, 3.0], name="a")
    view.set_y_label("m/s")
    view.redraw()
    figure.canvas.draw()
    assert view.ax.yaxis.label.get_fontsize() == _Y_LABEL
    assert view.ax.xaxis.label.get_fontsize() > _Y_LABEL
    assert view.ax.yaxis.get_major_ticks()[0].label1.get_fontsize() == _Y_TICK


def test_y_axis_name_clears_the_next_axis():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from linefold.mpl import MplView

    figure = Figure(figsize=(8, 4), dpi=100)
    FigureCanvasAgg(figure)
    view = MplView(figure=figure)
    shared = view.plot.add_line([1.0, 2.0, 3.0, 4.0], name="shared")
    inner = view.plot.add_line([10000.0, 25000.0, 18000.0, 31000.0], name="Channel pressure")
    outer = view.plot.add_line([0.12, 0.48, 0.33, 0.91], name="Displacement sensor")
    del shared
    view.plot.set_own_axis(inner, True)
    view.plot.set_own_axis(outer, True)
    view.set_y_label("millimetre")
    view.redraw()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    def column(axes):
        spine = axes.spines["left"].get_window_extent(renderer)
        label = axes.yaxis.label.get_window_extent(renderer)
        ticks = []
        for tick in axes.yaxis.get_major_ticks():
            box = tick.label1.get_window_extent(renderer)
            if tick.label1.get_text() and box.width > 1 and box.height > 1:
                ticks.append(box)
        return spine, label, ticks

    columns = [column(view.ax), *(column(axes) for axes in view._own)]
    for spine, label, ticks in columns:
        assert ticks
        if label.width > 1:
            assert label.x1 + 6 <= min(box.x0 for box in ticks)
    # columns run from the plot outward, so each name stays inside the next spine.
    for inner_col, outer_col in zip(columns, columns[1:]):
        _spine, label, ticks = inner_col
        outer_spine = outer_col[0]
        assert min(box.x0 for box in ticks) > outer_spine.x1 + 6
        if label.width > 1:
            assert label.x0 > outer_spine.x1 + 6


def test_base_e_tick_writes_the_letter():
    formatter = _ELogFormatter(math.e)
    for value in (math.e, math.e ** 2, 2 * math.e, math.e ** -1):
        text = formatter(value)
        assert "e" in text
        assert "2.718" not in text


def test_shift_on_the_wheel_does_not_need_a_click():
    class Event:
        def __init__(self, key, modifiers):
            self.key = key
            self.modifiers = modifiers

    assert "shift" in _held(Event(None, {"shift"}))
    assert "control" in _held(Event("control", ()))
    assert "ctrl" in _held(Event(None, {"ctrl"}))


def main():
    test_fair_split()
    test_chord_is_not_half()
    test_nan_is_not_an_extreme()
    test_time_base_bisects()
    test_log_edges_are_geometric()
    test_log_y_drops_non_positive()
    test_numpy_scan_matches_the_one_in_use()
    test_cached_envelope_matches_the_scan()
    test_the_numpy_ladder_matches_the_scan()
    test_panning_keeps_the_column()
    test_the_ladder_keeps_the_smaller_schedule()
    test_a_wide_column_uses_the_ladder()
    test_wheel_keeps_the_value_under_the_pointer()
    test_pan_and_reset_and_rect()
    test_project_vertical_and_time()
    test_close_zoom_joins_the_samples()
    test_log_axis_joins_sparse_columns()
    test_flat_log_front_stays_a_line()
    test_shift_claims_an_axis_and_puts_it_back()
    test_shared_time_leaves_a_private_axis()
    test_y_axis_text_is_smaller_than_the_x_axis()
    test_y_axis_name_clears_the_next_axis()
    test_base_e_tick_writes_the_letter()
    test_shift_on_the_wheel_does_not_need_a_click()
    print("ok")


if __name__ == "__main__":
    main()
