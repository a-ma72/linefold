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

A line of at least ``BASE`` samples keeps a ladder of those folds. Each rung
holds the first sample, the minimum, the maximum and the last sample of its
buckets, and a wide column is answered from the rungs that tile it. The chord
at a column edge is still the two raw samples beside that edge, so the picture
is the one the scan draws.
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


def _anchored_edges(x0, x1, columns, log_x):
    """Column edges on a grid that stays put while the window is panned.

    A column has the window's width, so the pane still gets one column a
    device pixel. The grid is anchored at zero — at one, on a logarithmic
    axis — and phased so each column holds the sample under that pixel's
    centre. A grid locked to the window would slide every boundary by a
    fraction of a sample as the pane moves, and the line would boil.
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
        k0 = math.floor(p0 / width + 0.5)
        return np.exp((k0 + np.arange(columns + 1, dtype=np.float64)) * width)
    width = (float(x1) - float(x0)) / columns
    if not width > 0.0:
        return None
    k0 = math.floor(float(x0) / width + 0.5)
    return (k0 + np.arange(columns + 1, dtype=np.float64)) * width


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


# A column narrower than this is cheaper to scan than to look up. The ladder
# starts here: one bucket is this many samples, aligned to a multiple of it.
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
    """``(nominal, starts, ends)`` from the coarsest rung down to the base.

    ``nominal`` is the aligned width. The last bucket of a rung is shorter
    when the line does not fill it, and it still starts on the alignment.
    """
    n = int(n)
    if n < BASE:
        return []
    count = (n + BASE - 1) // BASE
    starts = np.arange(count, dtype=np.int64) * BASE
    ends = np.minimum(starts + BASE, n)
    fine = [(BASE, starts, ends)]
    step = 0
    while fine[-1][1].size > 1:
        fan = _fan_at(pattern, step)
        step += 1
        child_nominal, child_starts, child_ends = fine[-1]
        nominal = child_nominal * fan
        first, last = _group_index(int(child_starts.size), fan)
        fine.append((nominal, child_starts[first], child_ends[last]))
    return list(reversed(fine))


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
    multiple of the base width, so the samples in it are one slice.
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
    base = BASE
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
    than a factor of two and stores no more than a factor of four. The small
    rungs dominate the store, so the extra large sizes the taper keeps are the
    whole of the difference.
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
    """The ladder of one line: four values a bucket, from ``BASE`` samples up.

    Built once. A span asks for the coarsest bucket that starts where the
    cursor is and ends inside the span, and reads raw samples only until the
    next alignment.
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
        self._linear = self._positive = None
        if int(data.size) < BASE:
            return
        geometry = _rungs(int(data.size), self.pattern)
        # Geometry is coarse-first. The values are folded fine-first, because
        # a rung is the min and the max of the rung below it.
        base = self._base_rung(geometry[-1])
        fine = [base]
        # geometry[-1] is the base; the coarser rungs sit in front of it.
        # Rebuild the merges from the base so the values follow the same cuts.
        child = base
        # Walk the fan steps the geometry used: one merge per rung above the base.
        for _level in range(len(geometry) - 1):
            fan = _fan_at(self.pattern, len(fine) - 1)
            child = _merge_rung(child, fan)
            fine.append(child)
        self.rungs = list(reversed(fine))
        self._linear, self._positive = _pack_rungs(self.rungs)

    def _base_rung(self, geometry):
        nominal, starts, ends = geometry
        rung = _Rung(nominal, starts, ends)
        _column_extents(self.data, starts, ends, rung.mins, rung.maxs, rung.counts, False)
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


def _pack_rungs(rungs):
    """One flat copy of every rung, coarse first, so one query can run in numba.

    The per-column walk used to pull buckets out of Python objects. A window
    of a long line is hundreds of columns, and that walk was the whole cost.
    """
    nominals = np.empty(len(rungs), dtype=np.int64)
    offsets = np.zeros(len(rungs) + 1, dtype=np.int64)
    for index, rung in enumerate(rungs):
        nominals[index] = rung.nominal
        offsets[index + 1] = offsets[index] + int(rung.starts.size)
    starts = np.concatenate([rung.starts for rung in rungs])
    ends = np.concatenate([rung.ends for rung in rungs])
    linear = (
        nominals, offsets, starts, ends,
        np.concatenate([rung.mins for rung in rungs]),
        np.concatenate([rung.maxs for rung in rungs]),
        np.concatenate([rung.counts for rung in rungs]),
    )
    positive = (
        nominals, offsets, starts, ends,
        np.concatenate([rung.pos_mins for rung in rungs]),
        np.concatenate([rung.pos_maxs for rung in rungs]),
        np.concatenate([rung.pos_counts for rung in rungs]),
    )
    return linear, positive


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
        data, starts, ends, mins, maxs, counts, positive_only,
        nominals, offsets, bucket_starts, bucket_ends, bucket_mins, bucket_maxs, bucket_counts,
    ):
        """Each column from the coarsest bucket that fits, raw only in the gaps.

        The gaps are the samples before the next multiple of the base width,
        at most that many, and they are folded here rather than one call per
        sample. A column shorter than the base is the plain scan: a bucket
        would be coarser than the column.
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
            else:
                at = a
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
                    # start before the next base alignment.
                    if at % base == 0:
                        nxt = at + base
                    else:
                        nxt = (at // base + 1) * base
                    if nxt > b:
                        nxt = b
                    lo, hi, seen = _fold_range(data, at, nxt, lo, hi, seen, positive_only)
                    at = nxt
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


def _extents_buckets(data, starts, ends, mins, maxs, counts, positive_only, rung):
    """Column extremes from the base buckets, with NumPy reductions only.

    A bucket that lies wholly inside a column contributes its minimum, its
    maximum and its count. What is left is the head before the next alignment
    and the tail after the last bucket that fit, gathered once and reduced
    with the same ``fmin`` the plain scan uses. The last bucket of the line
    may be shorter than the base width, so it is applied on its own.
    """
    base = int(rung.nominal)
    limit = int(data.size)
    starts = np.clip(np.asarray(starts, dtype=np.int64), 0, limit)
    ends = np.clip(np.asarray(ends, dtype=np.int64), 0, limit)
    if positive_only:
        bmin, bmax, bcount = rung.pos_mins, rung.pos_maxs, rung.pos_counts
    else:
        bmin, bmax, bcount = rung.mins, rung.maxs, rung.counts
    safe_min = np.where(bcount > 0, bmin, np.inf)
    safe_max = np.where(bcount > 0, bmax, -np.inf)
    n = int(starts.shape[0])
    nb = int(rung.starts.size)
    lo = np.full(n, np.inf, dtype=np.float64)
    hi = np.full(n, -np.inf, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int64)

    i0 = (starts + (base - 1)) // base
    # Every bucket except the last is exactly ``base`` long and ends on an
    # alignment. The slice [i0, i_full) is those whole buckets.
    i_full = np.minimum(ends // base, nb - 1)
    use = (i_full > i0) & (i0 < nb) & (i0 >= 0)
    if np.any(use):
        cols = np.flatnonzero(use)
        indices, bounds = _pieces(i0[cols], i_full[cols])
        piece_seen = np.add.reduceat(bcount[indices], bounds)
        _scatter_pieces(
            lo, hi, seen, cols,
            np.minimum.reduceat(safe_min[indices], bounds),
            np.maximum.reduceat(safe_max[indices], bounds),
            piece_seen,
        )

    last = nb - 1
    last_start = int(rung.starts[last])
    last_end = int(rung.ends[last])
    take_last = (i0 <= last) & (ends >= last_end) & (starts <= last_start)
    if np.any(take_last) and int(bcount[last]) > 0:
        lo[take_last] = np.minimum(lo[take_last], safe_min[last])
        hi[take_last] = np.maximum(hi[take_last], safe_max[last])
        seen[take_last] += int(bcount[last])

    head_hi = np.minimum(ends, i0 * base)
    # Where no whole bucket fit, the tail continues the head so the two
    # slices together are the column and nothing is read twice.
    tail_lo = np.where(take_last, last_end, np.where(use, i_full * base, head_hi))
    raw_starts = np.concatenate((starts, tail_lo))
    raw_ends = np.concatenate((head_hi, ends))
    raw_cols = np.concatenate((np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64)))
    keep = raw_ends > raw_starts
    if np.any(keep):
        indices, bounds = _pieces(raw_starts[keep], raw_ends[keep])
        block = data[indices]
        if positive_only:
            block = np.where(block > 0.0, block, np.nan)
        present = np.logical_not(np.isnan(block)).astype(np.int64)
        _scatter_pieces(
            lo, hi, seen, raw_cols[keep],
            np.fmin.reduceat(block, bounds),
            np.fmax.reduceat(block, bounds),
            np.add.reduceat(present, bounds),
        )

    good = (seen > 0) & (lo <= hi)
    mins[good] = lo[good]
    maxs[good] = hi[good]
    counts[good] = seen[good]


def _buckets_skip_the_fringe(starts, ends):
    """True when whole buckets hold more samples than the fringes around them.

    A fringe is at most two base widths, one at each end, and gathering it
    costs more per sample than one reduction straight through the column. The
    buckets are the smaller read once they cover more than that fringe.
    """
    widths = np.maximum(ends - starts, 0)
    inside = np.maximum(widths - 2 * BASE, 0)
    return int(inside.sum()) > int(widths.sum() - inside.sum())


def _extents(data, starts, ends, mins, maxs, counts, positive_only, cache):
    """The column scan, from the ladder where a column holds a base bucket.

    With numba the walk is one parallel pass. Without it, a column that holds
    several base buckets is a ``reduceat`` over those buckets plus one gathered
    reduction of the unaligned edges. Fewer buckets than that, and the plain
    reduction of the samples is the smaller read. A Python walk of the edges
    was the delay on a full window.
    """
    wide = cache is not None and cache.rungs and np.any(ends - starts >= BASE)
    if not wide:
        _column_extents(data, starts, ends, mins, maxs, counts, positive_only)
        return
    if _column_extents_ladder is not None:
        packed = cache._positive if positive_only else cache._linear
        _column_extents_ladder(
            cache.data, starts, ends, mins, maxs, counts, bool(positive_only), *packed,
        )
        return
    if _buckets_skip_the_fringe(starts, ends):
        _extents_buckets(
            cache.data, starts, ends, mins, maxs, counts, bool(positive_only), cache.rungs[-1],
        )
        return
    _column_extents(data, starts, ends, mins, maxs, counts, positive_only)


def envelope(data, x0, x1, columns, x=None, log_x=False, log_y=False, cache=None, anchored=False):
    """Min, max and sample count per column over ``[x0, x1)``.

    ``x`` is a monotonic time base, one value per sample. Without it, sample
    ``i`` is at ``i`` and the window is an index window. Returns mins, maxs,
    counts and the column edges (``columns + 1`` of them).

    ``cache`` is a :class:`ColumnCache` of this same array. A column of at
    least ``BASE`` samples is then read from it. The chord below still uses
    the samples on either side of each edge.

    ``anchored`` places the columns on :func:`_anchored_edges`, so a pan keeps
    each sample in the column it was in.
    """
    columns = int(columns)
    data = np.asarray(data)
    if data.dtype != np.float64 or not data.flags.c_contiguous:
        data = np.ascontiguousarray(data, dtype=np.float64)
    if anchored:
        edges = _anchored_edges(float(x0), float(x1), columns, log_x)
    else:
        edges = _edges(float(x0), float(x1), columns, log_x)
    mins = np.full(max(columns, 0), np.nan, dtype=np.float64)
    maxs = np.full(max(columns, 0), np.nan, dtype=np.float64)
    counts = np.zeros(max(columns, 0), dtype=np.int64)
    if edges is None:
        return mins, maxs, counts, np.empty(0, dtype=np.float64)

    samples = int(data.size)
    if x is None:
        starts, ends = _index_spans(edges, samples)
    else:
        x = np.asarray(x)
        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)
        starts, ends = _time_spans(x, edges)
    _extents(data, starts, ends, mins, maxs, counts, bool(log_y), cache)

    for column in range(columns):
        if x is None:
            _include(mins, maxs, column, y_on_chord(data, float(edges[column]), log_x, log_y))
            _include(mins, maxs, column, y_on_chord(data, float(edges[column + 1]), log_x, log_y))
        else:
            _include(mins, maxs, column, _chord_time(x, data, edges[column], log_x, log_y))
            _include(mins, maxs, column, _chord_time(x, data, edges[column + 1], log_x, log_y))
    return mins, maxs, counts, edges
