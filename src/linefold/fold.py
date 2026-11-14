# SPDX-FileCopyrightText: 2026 Andreas Martin
# SPDX-License-Identifier: BSD-2-Clause

"""Column envelope for one line.

This is the fold ``plot-algorithm.py`` was written to evaluate. A column keeps
the smallest and the largest sample that lands in it, in one pass. The chord
from the sample before a column edge to the sample after it crosses that edge
at a y that depends on where the edge sits, and that y lengthens each of the
two columns. Half is only the case where the edge is in the middle.

Sample ``i`` of an index line sits at ``x = i``. A monotonic time base is the
same split, read backwards: the edges are values, and the samples are the ones
whose time falls between them.

A line of at least ``BASE`` samples keeps a ladder of those folds. Rungs of
``FINE`` samples and up answer a column from the coarsest bucket that tiles
it, so a wide view does not walk every sample. From ``BASE`` up a bucket also
keeps its first sample, its last sample and the positive-only fold a
logarithmic y axis reads. The fine rungs below that keep only the minimum,
the maximum and the count. The chord at a column edge is still the two raw
samples beside that edge, so the picture is the one the scan draws.
"""

from __future__ import annotations

import math

import numpy as np

# Numba is an accelerator, not a requirement. The same three answers are
# computed in NumPy when it is absent: a NaN still fails both comparisons,
# and a value at or below zero is still left out of a logarithmic column.
try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range


def y_on_chord(data, x, log_x=False, log_y=False):
    """Y of the line between the samples on either side of an index ``x``.

    A logarithmic axis has no place for a value at or below zero, so that
    sample is a gap and the chord through it is not drawn.
    """
    last = data.size - 1
    if x < 0.0 or x > last:
        return None
    if x >= last:
        if log_x or log_y:
            value = float(data[last])
            if log_y and not value > 0.0:
                return None
            if log_x and not last > 0:
                return None
        return float(data[last])
    i_next = int(math.ceil(x))
    if i_next <= 0:
        value = float(data[0])
        if (log_x or log_y) and not value > 0.0:
            return None
        if log_x:
            return None
        return value
    i_prev = i_next - 1
    if log_x and not i_prev > 0:
        return None
    y_prev = float(data[i_prev])
    y_next = float(data[i_next])
    if y_prev != y_prev or y_next != y_next:
        return None
    if log_y and (not y_prev > 0.0 or not y_next > 0.0):
        return None
    t = (x - i_prev) / (i_next - i_prev)
    value = y_prev + t * (y_next - y_prev)
    if log_y and not value > 0.0:
        return None
    return value


def _include(mins, maxs, column, y):
    if y is None or y != y:
        return
    if mins[column] != mins[column]:
        mins[column] = y
        maxs[column] = y
    elif y < mins[column]:
        mins[column] = y
    elif y > maxs[column]:
        maxs[column] = y


def _column_extents_numpy(data, starts, ends, mins, maxs, counts, positive_only):
    """The scan without numba.

    The columns are one partition of the samples they cover, so one ``fmin``
    and one ``fmax`` over those slices is the same pair of comparisons. An
    empty column is left out: ``reduceat`` would otherwise keep the sample
    the empty slice starts on. A NaN is not a number to ``fmin``, which is
    the same drop the comparison loop gets because a NaN fails both tests.
    """
    n = int(starts.shape[0])
    if n == 0:
        return
    lengths = ends - starts
    partitioned = n == 1 or np.array_equal(ends[:-1], starts[1:])
    if not partitioned:
        # The columns are not one block, so reduceat cannot see them in order.
        # Gather each slice and reduce the pieces together.
        _reduce_gathered(data, starts, ends, mins, maxs, counts, positive_only)
        return
    valid = lengths > 0
    if not np.any(valid):
        return
    offset = int(starts[0])
    last = int(ends[-1])
    if last <= offset:
        return
    block = data[offset:last]
    if positive_only:
        block = np.where(block > 0.0, block, np.nan)
    bounds = starts[valid] - offset
    lo = np.fmin.reduceat(block, bounds)
    hi = np.fmax.reduceat(block, bounds)
    seen = np.add.reduceat(np.logical_not(np.isnan(block)).astype(np.int64), bounds)
    columns = np.flatnonzero(valid)
    keep = lo <= hi
    mins[columns[keep]] = lo[keep]
    maxs[columns[keep]] = hi[keep]
    counts[columns[keep]] = seen[keep]


def _monotonic_numpy(data):
    if data.shape[0] < 2:
        return True
    # A NaN makes the comparison false, so it is not a time base either.
    return bool(np.all(data[1:] >= data[:-1]))


def _limits_numpy(data):
    if data.shape[0] == 0:
        return np.inf, -np.inf, np.inf
    lo = np.fmin.reduce(data)
    hi = np.fmax.reduce(data)
    if not lo <= hi:
        lo = np.inf
        hi = -np.inf
    # +inf is above zero and not below the seed, so it is not a start for a
    # logarithmic axis. Only a finite positive value is.
    positive = np.isfinite(data) & (data > 0.0)
    if np.any(positive):
        pos = np.min(data, where=positive, initial=np.inf)
    else:
        pos = np.inf
    return lo, hi, pos


if njit is not None:

    @njit(parallel=True)
    def _column_extents(data, starts, ends, mins, maxs, counts, positive_only):
        """One pass, two comparisons a sample.

        A NaN fails both comparisons and drops out, so the walk needs no finiteness
        test of its own. The columns do not share samples, so they run side by side.
        """
        for c in prange(starts.shape[0]):
            a = starts[c]
            b = ends[c]
            if a >= b:
                continue
            lo = np.inf
            hi = -np.inf
            seen = 0
            for i in range(a, b):
                value = data[i]
                if positive_only and not (value > 0.0):
                    continue
                if value < lo:
                    lo = value
                if value > hi:
                    hi = value
                if value == value:
                    seen += 1
            if lo <= hi:
                mins[c] = lo
                maxs[c] = hi
                counts[c] = seen

    @njit()
    def monotonic(data):
        """True when each sample is at or after the one before it.

        A time base that doubles back is not a map that can be run backwards, so
        the fold keeps the index. A NaN is not ordered and fails the same test.
        """
        for i in range(1, data.shape[0]):
            if not (data[i] >= data[i - 1]):
                return False
        return True

    @njit()
    def limits(data):
        """Smallest, largest, and smallest strictly positive value.

        The third is where a logarithmic axis has to start. A value at or below
        zero is not a place on it, and it is left out of that end only.
        """
        lo = np.inf
        hi = -np.inf
        pos = np.inf
        for i in range(data.shape[0]):
            value = data[i]
            if value < lo:
                lo = value
            if value > hi:
                hi = value
            if value > 0.0 and value < pos:
                pos = value
        return lo, hi, pos

else:
    _column_extents = _column_extents_numpy
    monotonic = _monotonic_numpy
    limits = _limits_numpy


def _edges(x0, x1, columns, log_x):
    if columns < 1 or not (x1 > x0):
        return None
    if log_x:
        # Equal steps in ln. The base cancels, so the same edges come out of
        # any base above one.
        if not (x0 > 0.0):
            return None
        t = np.arange(columns + 1, dtype=np.float64) / columns
        log0 = math.log(x0)
        return np.exp(log0 + t * (math.log(x1) - log0))
    return x0 + np.arange(columns + 1, dtype=np.float64) * ((x1 - x0) / columns)


def _grid(x0, x1, columns, log_x):
    """``(width, origin, edges)`` of the anchored column grid, or None.

    ``origin`` is the integer index of the first edge. A later window with
    the same width and an origin an integer number of columns away is the
    same grid, slid.
    """
    columns = int(columns)
    if columns < 1 or not (x1 > x0):
        return None
    if log_x:
        if not (x0 > 0.0) or not (x1 > 0.0):
            return None
        p0 = math.log(x0)
        p1 = math.log(x1)
        width = (p1 - p0) / columns
        if not width > 0.0:
            return None
        origin = float(math.floor(p0 / width + 0.5))
        edges = np.exp((origin + np.arange(columns + 1, dtype=np.float64)) * width)
        return width, origin, edges
    width = (float(x1) - float(x0)) / columns
    if not width > 0.0:
        return None
    origin = float(math.floor(float(x0) / width + 0.5))
    edges = (origin + np.arange(columns + 1, dtype=np.float64)) * width
    return width, origin, edges


def _anchored_edges(x0, x1, columns, log_x):
    """Column edges on a grid that stays put while the window is panned.

    A column has the window's width, so the pane still gets one column a
    device pixel. The grid is anchored at zero — at one, on a logarithmic
    axis — and phased so each column holds the sample under that pixel's
    centre. A grid locked to the window would slide every boundary by a
    fraction of a sample as the pane moves, and the line would boil.
    """
    grid = _grid(x0, x1, columns, log_x)
    if grid is None:
        return None
    return grid[2]


def _index_spans(edges, samples):
    # Sample i belongs to the column that contains x = i. The hair under an
    # edge keeps a sample that sits on it in the column on the right, the
    # same closed form as splitting n samples into ``columns`` as evenly as
    # the integers allow.
    starts = np.ceil(edges[:-1] - 1e-9).astype(np.int64)
    ends = np.floor(edges[1:] - 1e-9).astype(np.int64) + 1
    np.clip(starts, 0, samples, out=starts)
    np.clip(ends, 0, samples, out=ends)
    return starts, ends


def _time_spans(x, edges):
    # The time base only ever goes one way, so a value is a position.
    starts = np.searchsorted(x, edges[:-1], side="left").astype(np.int64)
    ends = np.searchsorted(x, edges[1:], side="left").astype(np.int64)
    return starts, ends


def _chord_time(x, data, edge, log_x, log_y):
    i_prev = int(np.searchsorted(x, edge, side="right")) - 1
    i_next = i_prev + 1
    if i_prev < 0 or i_next >= x.size:
        return None
    x_prev = float(x[i_prev])
    x_next = float(x[i_next])
    if log_x and (not x_prev > 0.0 or not x_next > 0.0):
        return None
    if not x_next > x_prev:
        return None
    y_prev = float(data[i_prev])
    y_next = float(data[i_next])
    if y_prev != y_prev or y_next != y_next:
        return None
    if log_y and (not y_prev > 0.0 or not y_next > 0.0):
        return None
    t = (float(edge) - x_prev) / (x_next - x_prev)
    if t < 0.0 or t > 1.0:
        return None
    value = y_prev + t * (y_next - y_prev)
    if log_y and not value > 0.0:
        return None
    return value


# The finest bucket a column is answered from. Anything shorter is a handful
# of samples, and it is cheaper to read them than to look a bucket up. The
# rungs below ``BASE`` are this width doubled twice over, and they store only
# the minimum, the maximum and the count.
FINE = 4

# One bucket of the stored ladder. The rungs from here up keep the positive
# copy a logarithmic y reads, and the first and last sample of the bucket.
# A column shorter than this used to be a raw scan; the fine rungs are what
# answers it.
BASE = 1024

# How many buckets of one rung become one bucket of the next. The last entry
# repeats, so a pattern of one number is that factor all the way up. The
# taper groups the small buckets by fours, then by threes, and the large ones
# by twos: the small rungs are where the buckets are, and a large rung wants
# a size between the powers so a span just short of one does not shatter.
FAN_FOUR = (4,)
FAN_TWO = (2,)
FAN_TAPER = (4, 3, 2)


def _fan_at(pattern, step):
    if step < len(pattern):
        return int(pattern[step])
    return int(pattern[-1])


def _rungs(n, pattern):
    """``(nominal, starts, ends)`` from the coarsest rung down to ``FINE``.

    ``nominal`` is the aligned width. The last bucket of a rung is shorter
    when the line does not fill it, and it still starts on the alignment.
    Up to ``BASE`` the fan is four, so the fine rungs are 4, 16, 64 and 256.
    Above ``BASE`` the fan is ``pattern``.
    """
    n = int(n)
    if n < BASE:
        return []
    count = (n + FINE - 1) // FINE
    starts = np.arange(count, dtype=np.int64) * FINE
    ends = np.minimum(starts + FINE, n)
    built = [(FINE, starts, ends)]
    while built[-1][0] < BASE:
        child_nominal, child_starts, child_ends = built[-1]
        first, last = _group_index(int(child_starts.size), 4)
        built.append((child_nominal * 4, child_starts[first], child_ends[last]))
    step = 0
    while built[-1][1].size > 1:
        fan = _fan_at(pattern, step)
        step += 1
        child_nominal, child_starts, child_ends = built[-1]
        first, last = _group_index(int(child_starts.size), fan)
        built.append((child_nominal * fan, child_starts[first], child_ends[last]))
    return list(reversed(built))


def _rung_parts(rung):
    if isinstance(rung, tuple):
        return rung
    return rung.nominal, rung.starts, rung.ends


def _group_index(count, fan):
    """First and last child of each group of ``fan``, the tail group shorter."""
    groups = (int(count) + fan - 1) // fan
    first = np.arange(groups, dtype=np.int64) * fan
    last = np.minimum(first + fan, int(count)) - 1
    return first, last


def _advance(rungs, at, end):
    """The coarsest bucket that starts at ``at`` and ends by ``end``.

    None when no bucket starts here and still fits. The gap runs to the next
    multiple of ``FINE``, so the samples in it are one slice.
    """
    for level, rung in enumerate(rungs):
        nominal, starts, ends = _rung_parts(rung)
        if at % nominal != 0:
            continue
        index = at // nominal
        if index >= starts.size or int(starts[index]) != at:
            continue
        span = int(ends[index]) - at
        if span > 0 and at + span <= end:
            return level, index, span
    return None


def _bucket_count(n, pattern):
    return sum(int(_rung_parts(rung)[1].size) for rung in _rungs(n, pattern))


def _bucket_reads(n, pattern, start, end):
    """Cached buckets a span consumes. Samples between alignments are not buckets."""
    rungs = _rungs(n, pattern)
    at = int(start)
    end = int(end)
    reads = 0
    base = FINE
    while at < end:
        step = _advance(rungs, at, end)
        if step is None:
            nxt = at + base if at % base == 0 else (at // base + 1) * base
            at = end if nxt > end else nxt
            continue
        _level, _index, span = step
        at += span
        reads += 1
    return reads


def _probe_spans(n):
    """Spans a column of a long line actually asks for, including a short remainder."""
    spans = (
        (0, n),
        (0, min(n, max(BASE, n // 64))),
        (0, min(n, 65536 - 1)),
        (BASE // 2, max(BASE // 2 + 1, n - BASE // 2)),
        (BASE, min(n, BASE + 50000)),
        (0, min(n, 12288 + 100)),
    )
    return [(int(start), int(end)) for start, end in spans if end > start]


def schedule_cost(n=1 << 20):
    """Stored buckets, and buckets read over :func:`_probe_spans`, per schedule.

    Counts, not durations. The taper is the one to keep when it reads no more
    than the twos and stores no more than the fours. The fine rungs are in
    that store, and they are the same four-wide ladder for every schedule, so
    what still decides it is the rungs above ``BASE``.
    """
    patterns = {"four": FAN_FOUR, "two": FAN_TWO, "taper": FAN_TAPER}
    stored = {name: _bucket_count(n, pattern) for name, pattern in patterns.items()}
    spans = _probe_spans(n)
    reads = {
        name: sum(_bucket_reads(n, pattern, start, end) for start, end in spans)
        for name, pattern in patterns.items()
    }
    return stored, reads


def chosen_pattern(n=1 << 20):
    """The schedule the ladder is built with. See :func:`schedule_cost`."""
    stored, reads = schedule_cost(n)
    if reads["taper"] <= reads["two"] and stored["taper"] <= stored["four"]:
        return FAN_TAPER
    return FAN_FOUR


PATTERN = chosen_pattern()


class _SlimRung:
    """One fine rung: minimum, maximum and count. No first, last or positive copy.

    The rungs below ``BASE`` are most of the ladder's memory. A logarithmic y
    does not read them: that axis still wants the positive copy, which the
    rungs from ``BASE`` up keep.
    """

    __slots__ = ("nominal", "starts", "ends", "mins", "maxs", "counts")

    def __init__(self, nominal, starts, ends):
        self.nominal = int(nominal)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = np.asarray(ends, dtype=np.int64)
        count = int(self.starts.size)
        self.mins = np.full(count, np.nan, dtype=np.float64)
        self.maxs = np.full(count, np.nan, dtype=np.float64)
        self.counts = np.zeros(count, dtype=np.int64)


def _merge_slim(child, fan):
    """One bucket per ``fan`` children, minimum and maximum and count only."""
    count = int(child.starts.size)
    fan = int(fan)
    first, last = _group_index(count, fan)
    groups = int(first.size)
    rung = _SlimRung(child.nominal * fan, child.starts[first].copy(), child.ends[last].copy())
    rung.counts[:] = _group_sum(child.counts, count, groups, fan)
    _group_minmax(
        rung.mins, rung.maxs, rung.counts, child.mins, child.maxs, child.counts,
        count, groups, fan,
    )
    return rung


class _Rung:
    __slots__ = (
        "nominal", "starts", "ends", "entry", "exit", "mins", "maxs", "counts",
        "pos_mins", "pos_maxs", "pos_counts",
    )

    def __init__(self, nominal, starts, ends):
        self.nominal = int(nominal)
        self.starts = starts
        self.ends = ends
        count = int(starts.size)
        self.entry = np.empty(count, dtype=np.float64)
        self.exit = np.empty(count, dtype=np.float64)
        self.mins = np.full(count, np.nan, dtype=np.float64)
        self.maxs = np.full(count, np.nan, dtype=np.float64)
        self.counts = np.zeros(count, dtype=np.int64)
        self.pos_mins = np.full(count, np.nan, dtype=np.float64)
        self.pos_maxs = np.full(count, np.nan, dtype=np.float64)
        self.pos_counts = np.zeros(count, dtype=np.int64)


def _group_sum(values, count, groups, fan):
    """Sum each group of ``fan`` children. The tail group is padded with zeros."""
    padded = np.zeros(groups * fan, dtype=np.int64)
    padded[:count] = values
    return padded.reshape(groups, fan).sum(axis=1)


def _group_minmax(out_min, out_max, out_count, mins, maxs, counts, count, groups, fan):
    """Min and max of the children that hold a sample. An empty group stays empty."""
    width = groups * fan
    lo = np.full(width, np.inf, dtype=np.float64)
    hi = np.full(width, -np.inf, dtype=np.float64)
    held = counts > 0
    lo[:count] = np.where(held, mins, np.inf)
    hi[:count] = np.where(held, maxs, -np.inf)
    lo = lo.reshape(groups, fan).min(axis=1)
    hi = hi.reshape(groups, fan).max(axis=1)
    has = out_count > 0
    out_min[has] = lo[has]
    out_max[has] = hi[has]


def _merge_rung(child, fan):
    """One bucket per ``fan`` children. The tail is a short bucket, still exact.

    Entry is the first child's first sample and exit the last child's last, so
    the four values of the union are the four values of its parts. A child
    with no finite sample does not move the extremes. The groups are reshaped
    and reduced together: one minimum and one sum, not a pass per group.
    """
    count = int(child.starts.size)
    fan = int(fan)
    first, last = _group_index(count, fan)
    groups = int(first.size)
    rung = _Rung(child.nominal * fan, child.starts[first].copy(), child.ends[last].copy())
    rung.entry[:] = child.entry[first]
    rung.exit[:] = child.exit[last]
    rung.counts[:] = _group_sum(child.counts, count, groups, fan)
    rung.pos_counts[:] = _group_sum(child.pos_counts, count, groups, fan)
    _group_minmax(
        rung.mins, rung.maxs, rung.counts, child.mins, child.maxs, child.counts,
        count, groups, fan,
    )
    _group_minmax(
        rung.pos_mins, rung.pos_maxs, rung.pos_counts,
        child.pos_mins, child.pos_maxs, child.pos_counts, count, groups, fan,
    )
    return rung


class ColumnCache:
    """The ladder of one line, from ``FINE`` samples up.

    Built once. A span asks for the coarsest bucket that starts where the
    cursor is and ends inside the span, and reads raw samples only until the
    next multiple of ``FINE``. The rungs below ``BASE`` are the slim ones.
    From ``BASE`` up each bucket also keeps the positive-only fold.
    """

    def __init__(self, data, pattern=None):
        data = np.asarray(data)
        if data.dtype != np.float64 or not data.flags.c_contiguous:
            data = np.ascontiguousarray(data, dtype=np.float64)
        self.data = data
        self.pattern = FAN_FOUR if pattern is None else tuple(pattern)
        if pattern is None:
            self.pattern = PATTERN
        self.rungs = []
        self.fine = []
        self._linear = self._positive = None
        if int(data.size) < BASE:
            return
        geometry = _rungs(int(data.size), self.pattern)
        base_at = next(index for index, rung in enumerate(geometry) if int(rung[0]) == BASE)
        # geometry is coarse-first, so the fine rungs sit behind the base and
        # the last of them is ``FINE``. Fold that one from the samples, then
        # merge by fours up to the base. The base's own minimum and maximum
        # are that merge; its positive copy is a scan, which is the only
        # positive fold the ladder keeps.
        slim = _SlimRung(*geometry[-1])
        _column_extents(self.data, slim.starts, slim.ends, slim.mins, slim.maxs, slim.counts, False)
        slim_levels = [slim]
        while slim_levels[-1].nominal < BASE:
            slim_levels.append(_merge_slim(slim_levels[-1], 4))
        self.fine = list(reversed(slim_levels[:-1]))
        base = self._base_rung(geometry[base_at], slim_levels[-1])
        built = [base]
        for step in range(base_at):
            built.append(_merge_rung(built[-1], _fan_at(self.pattern, step)))
        self.rungs = list(reversed(built))
        self._linear = _pack_fields(self.rungs + self.fine, False)
        self._positive = _pack_fields(self.rungs, True)

    def _base_rung(self, geometry, linear):
        nominal, starts, ends = geometry
        rung = _Rung(nominal, starts, ends)
        rung.mins[:] = linear.mins
        rung.maxs[:] = linear.maxs
        rung.counts[:] = linear.counts
        _column_extents(
            self.data, starts, ends, rung.pos_mins, rung.pos_maxs, rung.pos_counts, True,
        )
        rung.entry[:] = self.data[starts]
        rung.exit[:] = self.data[ends - 1]
        return rung

    def span(self, start, end, positive_only=False):
        """Minimum, maximum and finite count of samples ``[start, end)``."""
        mins = np.full(1, np.nan, dtype=np.float64)
        maxs = np.full(1, np.nan, dtype=np.float64)
        counts = np.zeros(1, dtype=np.int64)
        starts = np.asarray([int(start)], dtype=np.int64)
        ends = np.asarray([int(end)], dtype=np.int64)
        _extents(self.data, starts, ends, mins, maxs, counts, bool(positive_only), self)
        if int(counts[0]) <= 0:
            return math.nan, math.nan, 0
        return float(mins[0]), float(maxs[0]), int(counts[0])


def _pack_fields(rungs, positive):
    """One flat copy of every rung, coarse first, so one query can run in numba.

    The per-column walk used to pull buckets out of Python objects. A window
    of a long line is hundreds of columns, and that walk was the whole cost.
    ``positive`` selects the positive-only copy, which only the rungs from
    ``BASE`` up carry.
    """
    nominals = np.empty(len(rungs), dtype=np.int64)
    offsets = np.zeros(len(rungs) + 1, dtype=np.int64)
    for index, rung in enumerate(rungs):
        nominals[index] = rung.nominal
        offsets[index + 1] = offsets[index] + int(rung.starts.size)
    if positive:
        mins = np.concatenate([rung.pos_mins for rung in rungs])
        maxs = np.concatenate([rung.pos_maxs for rung in rungs])
        counts = np.concatenate([rung.pos_counts for rung in rungs])
    else:
        mins = np.concatenate([rung.mins for rung in rungs])
        maxs = np.concatenate([rung.maxs for rung in rungs])
        counts = np.concatenate([rung.counts for rung in rungs])
    return (
        nominals, offsets,
        np.concatenate([rung.starts for rung in rungs]),
        np.concatenate([rung.ends for rung in rungs]),
        mins, maxs, counts,
    )


if njit is not None:

    @njit(cache=True)
    def _fold_range(data, a, b, lo, hi, seen, positive_only):
        """The samples ``[a, b)``, on the same terms as ``_column_extents``."""
        for i in range(a, b):
            value = data[i]
            if positive_only and not (value > 0.0):
                continue
            if value < lo:
                lo = value
            if value > hi:
                hi = value
            if value == value:
                seen += 1
        return lo, hi, seen

    @njit(parallel=True, cache=True)
    def _column_extents_ladder(
        data, starts, ends, mins, maxs, counts, positive_only, reads,
        nominals, offsets, bucket_starts, bucket_ends, bucket_mins, bucket_maxs, bucket_counts,
    ):
        """Each column from the coarsest bucket that fits, raw only in the gaps.

        The gaps are the samples before the next multiple of the finest width,
        at most that many, and they are folded here rather than one call per
        sample. A column shorter than that finest rung is the plain scan: no
        bucket would fit inside it. ``reads`` counts those raw samples, not
        the buckets and not the chord.
        """
        base = nominals[nominals.shape[0] - 1]
        nlevels = nominals.shape[0]
        for c in prange(starts.shape[0]):
            a = starts[c]
            b = ends[c]
            if a >= b:
                continue
            lo = np.inf
            hi = -np.inf
            seen = np.int64(0)
            if b - a < base:
                lo, hi, seen = _fold_range(data, a, b, lo, hi, seen, positive_only)
                reads[c] = b - a
            else:
                at = a
                raw = np.int64(0)
                while at < b:
                    taken = np.int64(0)
                    for level in range(nlevels):
                        nominal = nominals[level]
                        if at % nominal != 0:
                            continue
                        index = at // nominal
                        begin = offsets[level]
                        if index < 0 or index >= offsets[level + 1] - begin:
                            continue
                        pos = begin + index
                        if bucket_starts[pos] != at:
                            continue
                        span = bucket_ends[pos] - at
                        if span > 0 and at + span <= b:
                            cnt = bucket_counts[pos]
                            if cnt > 0:
                                if bucket_mins[pos] < lo:
                                    lo = bucket_mins[pos]
                                if bucket_maxs[pos] > hi:
                                    hi = bucket_maxs[pos]
                                seen += cnt
                            taken = span
                            break
                    if taken > 0:
                        at += taken
                        continue
                    # No bucket starts here and still fits. The next one cannot
                    # start before the next alignment of the finest rung.
                    if at % base == 0:
                        nxt = at + base
                    else:
                        nxt = (at // base + 1) * base
                    if nxt > b:
                        nxt = b
                    lo, hi, seen = _fold_range(data, at, nxt, lo, hi, seen, positive_only)
                    raw += nxt - at
                    at = nxt
                reads[c] = raw
            if lo <= hi:
                mins[c] = lo
                maxs[c] = hi
                counts[c] = seen

else:
    _column_extents_ladder = None


def _reduce_gathered(data, starts, ends, mins, maxs, counts, positive_only):
    """Min, max and count of slices that are not one contiguous block.

    Each slice is gathered, the pieces are reduced together, and a slice with
    no finite sample is left as the caller filled it.
    """
    keep = ends > starts
    if not np.any(keep):
        return
    columns = np.flatnonzero(keep)
    indices, bounds = _pieces(starts[columns], ends[columns])
    block = data[indices]
    if positive_only:
        block = np.where(block > 0.0, block, np.nan)
    present = np.logical_not(np.isnan(block)).astype(np.int64)
    lo = np.fmin.reduceat(block, bounds)
    hi = np.fmax.reduceat(block, bounds)
    seen = np.add.reduceat(present, bounds)
    good = (seen > 0) & (lo <= hi)
    mins[columns[good]] = lo[good]
    maxs[columns[good]] = hi[good]
    counts[columns[good]] = seen[good]


def _pieces(starts, ends):
    """Indices of every ``[starts[i], ends[i])``, laid end to end.

    ``bounds`` is where each piece starts in that row, so one ``reduceat``
    there is one reduction a piece. The pieces may sit anywhere in the source
    and may overlap; gathering them is what keeps the reduction off a Python
    walk of the columns.
    """
    lengths = np.asarray(ends - starts, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    bounds = np.cumsum(lengths) - lengths
    within = np.arange(total, dtype=np.int64) - np.repeat(bounds, lengths)
    return np.repeat(starts, lengths) + within, bounds


def _scatter_pieces(lo, hi, seen, columns, piece_lo, piece_hi, piece_seen):
    """Fold piece reductions into columns. Two pieces may name one column."""
    ok = piece_seen > 0
    if not np.any(ok):
        return
    columns = columns[ok]
    np.minimum.at(lo, columns, piece_lo[ok])
    np.maximum.at(hi, columns, piece_hi[ok])
    np.add.at(seen, columns, piece_seen[ok])


def _scatter_raw(data, cols, starts, ends, lo, hi, seen, reads, positive_only):
    """Fold raw slices into columns and count every sample they walk."""
    if int(cols.size) == 0:
        return
    keep = ends > starts
    if not np.any(keep):
        return
    cols = cols[keep]
    starts = starts[keep]
    ends = ends[keep]
    np.add.at(reads, cols, ends - starts)
    indices, bounds = _pieces(starts, ends)
    block = data[indices]
    if positive_only:
        block = np.where(block > 0.0, block, np.nan)
    present = np.logical_not(np.isnan(block)).astype(np.int64)
    _scatter_pieces(
        lo, hi, seen, cols,
        np.fmin.reduceat(block, bounds),
        np.fmax.reduceat(block, bounds),
        np.add.reduceat(present, bounds),
    )


def _eat_rung(rung, cols, starts, ends, lo, hi, seen, positive_only):
    """Cover whole buckets of one rung. What is left is the head and the tail.

    A bucket that does not fit stays in the leftover, for a finer rung or for
    the raw scan. The leftover of a fragment the rung does not touch is that
    fragment, once, so a short span is not doubled at every level.
    """
    if positive_only:
        bmin, bmax, bcount = rung.pos_mins, rung.pos_maxs, rung.pos_counts
    else:
        bmin, bmax, bcount = rung.mins, rung.maxs, rung.counts
    width = int(rung.nominal)
    count = int(rung.starts.size)
    align = ((starts + (width - 1)) // width) * width
    i0 = align // width
    i_lim = np.minimum(ends // width, count - 1)
    use = (i_lim > i0) & (i0 >= 0) & (i0 < count)
    safe_min = np.where(bcount > 0, bmin, np.inf)
    safe_max = np.where(bcount > 0, bmax, -np.inf)
    if np.any(use):
        chosen = np.flatnonzero(use)
        indices, bounds = _pieces(i0[chosen], i_lim[chosen])
        _scatter_pieces(
            lo, hi, seen, cols[chosen],
            np.minimum.reduceat(safe_min[indices], bounds),
            np.maximum.reduceat(safe_max[indices], bounds),
            np.add.reduceat(bcount[indices], bounds),
        )
    last = count - 1
    last_start = int(rung.starts[last])
    last_end = int(rung.ends[last])
    take_last = (i0 <= last) & (ends >= last_end) & (starts <= last_start)
    if np.any(take_last) and int(bcount[last]) > 0:
        chosen = cols[take_last]
        lo[chosen] = np.minimum(lo[chosen], safe_min[last])
        hi[chosen] = np.maximum(hi[chosen], safe_max[last])
        seen[chosen] += int(bcount[last])
    cover_lo = np.where(use, align, ends)
    cover_hi = np.where(use, i_lim.astype(np.int64) * width, cover_lo)
    cover_lo = np.where(take_last & np.logical_not(use), last_start, cover_lo)
    cover_hi = np.where(take_last, np.maximum(cover_hi, last_end), cover_hi)
    return (
        np.concatenate((cols, cols)),
        np.concatenate((starts, np.maximum(cover_hi, starts))),
        np.concatenate((np.minimum(cover_lo, ends), ends)),
    )


def _extents_rungs(data, starts, ends, mins, maxs, counts, positive_only, rungs, reads):
    """The NumPy ladder. Same buckets as the parallel walk, reduced in arrays.

    A column shorter than the finest rung is read. Every coarser bucket that
    fits is taken, and the samples no bucket covered are the only raw read.
    """
    n = int(starts.shape[0])
    lo = np.full(n, np.inf, dtype=np.float64)
    hi = np.full(n, -np.inf, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int64)
    finest = int(rungs[-1].nominal)
    index = np.arange(n, dtype=np.int64)
    short = (ends - starts) < finest
    if np.any(short):
        _scatter_raw(data, index[short], starts[short], ends[short], lo, hi, seen, reads, positive_only)
    live = (ends - starts) >= finest
    cols = index[live]
    left = starts[live]
    right = ends[live]
    for rung in rungs:
        if int(cols.size) == 0:
            break
        cols, left, right = _eat_rung(rung, cols, left, right, lo, hi, seen, positive_only)
        keep = right > left
        cols = cols[keep]
        left = left[keep]
        right = right[keep]
    if int(cols.size):
        _scatter_raw(data, cols, left, right, lo, hi, seen, reads, positive_only)
    good = (seen > 0) & (lo <= hi)
    mins[good] = lo[good]
    maxs[good] = hi[good]
    counts[good] = seen[good]



def _near(left, right):
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 1e-9 * scale


def _extents(data, starts, ends, mins, maxs, counts, positive_only, cache):
    """The column scan. Returns how many raw samples it walked.

    A column at least as wide as the finest rung of the ladder in use is
    answered from that ladder. On a linear axis the finest rung is ``FINE``,
    so a wide view reads the unaligned fringe and not the column. On a
    logarithmic y the finest rung is ``BASE``: the fine rungs do not separate
    a positive sample, and a column shorter than ``BASE`` is read. With numba
    the walk is one parallel pass. Without it, the same buckets are reduced
    with ``reduceat``. The chord is not part of this count.
    """
    reads = np.zeros(int(starts.shape[0]), dtype=np.int64)
    finest = BASE if positive_only else FINE
    use_ladder = (
        cache is not None and cache.rungs
        and np.any(ends - starts >= finest)
    )
    if not use_ladder:
        _column_extents(data, starts, ends, mins, maxs, counts, positive_only)
        span = np.maximum(ends - starts, 0)
        return int(span.sum())
    packed = cache._positive if positive_only else cache._linear
    if _column_extents_ladder is not None:
        _column_extents_ladder(
            cache.data, starts, ends, mins, maxs, counts, bool(positive_only), reads, *packed,
        )
        return int(reads.sum())
    rungs = cache.rungs if positive_only else list(cache.rungs) + list(cache.fine)
    _extents_rungs(
        cache.data, starts, ends, mins, maxs, counts, bool(positive_only), rungs, reads,
    )
    return int(reads.sum())


def _shift_into(values, shift):
    """Slide ``values`` by whole columns. Positive drops the left."""
    if shift > 0:
        values[:-shift] = values[shift:].copy()
    elif shift < 0:
        step = -shift
        values[step:] = values[:-step].copy()


def _apply_chords(data, x, edges, mins, maxs, first, last, log_x, log_y):
    for column in range(int(first), int(last)):
        left = float(edges[column])
        right = float(edges[column + 1])
        if x is None:
            _include(mins, maxs, column, y_on_chord(data, left, log_x, log_y))
            _include(mins, maxs, column, y_on_chord(data, right, log_x, log_y))
        else:
            _include(mins, maxs, column, _chord_time(x, data, left, log_x, log_y))
            _include(mins, maxs, column, _chord_time(x, data, right, log_x, log_y))


class ColumnScratch:
    """The columns of one line, kept so a pan can slide them.

    A pan by whole columns moves each sample with the column it was already
    in. The columns that stay are slid, and only the strip that entered is
    walked. A pan that does not cross a column walks nothing. Zoom, a
    different width, a log change or a new array fills every column again.
    """

    __slots__ = (
        "mins", "maxs", "counts", "valid", "columns", "width", "origin",
        "log_x", "log_y", "data_id", "count", "first", "last", "x_id",
        "x_first", "x_last", "samples_read", "columns_filled",
    )

    def __init__(self):
        self.mins = None
        self.maxs = None
        self.counts = None
        self.valid = False
        self.columns = 0
        self.width = 0.0
        self.origin = 0.0
        self.log_x = False
        self.log_y = False
        self.data_id = None
        self.count = 0
        self.first = 0.0
        self.last = 0.0
        self.x_id = None
        self.x_first = 0.0
        self.x_last = 0.0
        self.samples_read = 0
        self.columns_filled = 0

    def drop(self):
        self.valid = False

    def prepare(self, columns):
        columns = int(columns)
        self.mins = np.full(columns, np.nan, dtype=np.float64)
        self.maxs = np.full(columns, np.nan, dtype=np.float64)
        self.counts = np.zeros(columns, dtype=np.int64)
        self.columns = columns
        self.valid = False

    def shift_of(self, data, x, columns, width, origin, log_x, log_y):
        """Whole columns the window moved, or None when nothing can slide."""
        if not self.valid or self.columns != int(columns):
            return None
        if self.log_x != bool(log_x) or self.log_y != bool(log_y):
            return None
        if self.data_id != id(data) or self.count != int(data.size) or int(data.size) == 0:
            return None
        if not _near(self.first, float(data[0])) or not _near(self.last, float(data[-1])):
            return None
        x_id = None if x is None else id(x)
        if self.x_id != x_id:
            return None
        if x is not None and (
            int(x.size) == 0
            or not _near(self.x_first, float(x[0]))
            or not _near(self.x_last, float(x[-1]))
        ):
            return None
        if not _near(self.width, float(width)):
            return None
        shifted = float(origin) - float(self.origin)
        shift = int(round(shifted))
        if abs(shifted - float(shift)) > 1e-6 or abs(shift) >= int(columns):
            return None
        return shift

    def remember(self, data, x, width, origin, log_x, log_y):
        self.valid = True
        self.data_id = id(data)
        self.count = int(data.size)
        self.first = float(data[0])
        self.last = float(data[-1])
        self.x_id = None if x is None else id(x)
        if x is None:
            self.x_first = 0.0
            self.x_last = 0.0
        else:
            self.x_first = float(x[0])
            self.x_last = float(x[-1])
        self.width = float(width)
        self.origin = float(origin)
        self.log_x = bool(log_x)
        self.log_y = bool(log_y)


def envelope(
    data, x0, x1, columns, x=None, log_x=False, log_y=False, cache=None, anchored=False,
    scratch=None,
):
    """Min, max and sample count per column over ``[x0, x1)``.

    ``x`` is a monotonic time base, one value per sample. Without it, sample
    ``i`` is at ``i`` and the window is an index window. Returns mins, maxs,
    counts and the column edges (``columns + 1`` of them).

    ``cache`` is a :class:`ColumnCache` of this same array. A linear column of
    at least ``FINE`` samples is read from it, and a logarithmic column of at
    least ``BASE`` samples. The chord still uses the samples on either side
    of each edge, and it is not part of :attr:`samples_read`.

    ``anchored`` places the columns on :func:`_anchored_edges`, so a pan keeps
    each sample in the column it was in. ``scratch`` is a :class:`ColumnScratch`
    kept for this line. A later pan by whole columns slides it and fills only
    the strip that entered. :attr:`samples_read` is the raw samples that strip
    walked, and :attr:`columns_filled` is how many columns were walked.
    """
    columns = int(columns)
    log_x = bool(log_x)
    log_y = bool(log_y)
    anchored = bool(anchored)
    data = np.asarray(data)
    if data.dtype != np.float64 or not data.flags.c_contiguous:
        data = np.ascontiguousarray(data, dtype=np.float64)
    if x is not None:
        x = np.asarray(x)
        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)
    width = origin = None
    if anchored:
        grid = _grid(float(x0), float(x1), columns, log_x)
        edges = None if grid is None else grid[2]
        if grid is not None:
            width, origin = grid[0], grid[1]
    else:
        edges = _edges(float(x0), float(x1), columns, log_x)

    def finish(reads, filled, out_mins, out_maxs, out_counts, out_edges):
        envelope.samples_read = int(reads)
        envelope.columns_filled = int(filled)
        if scratch is not None:
            scratch.samples_read = int(reads)
            scratch.columns_filled = int(filled)
        return out_mins, out_maxs, out_counts, out_edges

    if edges is None:
        empty = max(columns, 0)
        if scratch is not None:
            scratch.drop()
        return finish(
            0, 0,
            np.full(empty, np.nan, dtype=np.float64),
            np.full(empty, np.nan, dtype=np.float64),
            np.zeros(empty, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    samples = int(data.size)
    if x is None:
        starts, ends = _index_spans(edges, samples)
    else:
        starts, ends = _time_spans(x, edges)

    shift = None
    if scratch is not None and anchored and width is not None and samples:
        shift = scratch.shift_of(data, x, columns, width, origin, log_x, log_y)
    if shift is not None:
        if shift != 0:
            _shift_into(scratch.mins, shift)
            _shift_into(scratch.maxs, shift)
            _shift_into(scratch.counts, shift)
        if shift > 0:
            fill_from = columns - shift
            fill_to = columns
        elif shift < 0:
            fill_from = 0
            fill_to = -shift
        else:
            fill_from = 0
            fill_to = 0
        if fill_to > fill_from:
            scratch.mins[fill_from:fill_to] = np.nan
            scratch.maxs[fill_from:fill_to] = np.nan
            scratch.counts[fill_from:fill_to] = 0
            reads = _extents(
                data, starts[fill_from:fill_to], ends[fill_from:fill_to],
                scratch.mins[fill_from:fill_to], scratch.maxs[fill_from:fill_to],
                scratch.counts[fill_from:fill_to], log_y, cache,
            )
            _apply_chords(
                data, x, edges, scratch.mins, scratch.maxs, fill_from, fill_to, log_x, log_y,
            )
        else:
            reads = 0
        scratch.remember(data, x, width, origin, log_x, log_y)
        return finish(reads, fill_to - fill_from, scratch.mins, scratch.maxs, scratch.counts, edges)

    if scratch is not None:
        scratch.prepare(columns)
        mins, maxs, counts = scratch.mins, scratch.maxs, scratch.counts
    else:
        mins = np.full(columns, np.nan, dtype=np.float64)
        maxs = np.full(columns, np.nan, dtype=np.float64)
        counts = np.zeros(columns, dtype=np.int64)
    reads = _extents(data, starts, ends, mins, maxs, counts, log_y, cache)
    _apply_chords(data, x, edges, mins, maxs, 0, columns, log_x, log_y)
    if scratch is not None and anchored and width is not None and samples:
        scratch.remember(data, x, width, origin, log_x, log_y)
    elif scratch is not None:
        scratch.drop()
    return finish(reads, columns, mins, maxs, counts, edges)


envelope.samples_read = 0
envelope.columns_filled = 0
