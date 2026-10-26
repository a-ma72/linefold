# SPDX-FileCopyrightText: 2026 Andreas Martin
# SPDX-License-Identifier: BSD-2-Clause

"""Lines, the camera, and the column envelope.

``MplView`` and the RPC3 catalog call this. The picture is the column
envelope from ``fold``: the extremes of a column, then the chord at each
edge. A long line keeps that fold in a ladder, so a wide column does not
walk its samples again. The value under the pointer stays put, a quarter of
the window stays over the data, and a logarithmic base only numbers the powers.
"""

from __future__ import annotations

import math

import numpy as np

from .fold import BASE, ColumnCache, _chord_time, envelope, limits, monotonic, y_on_chord

# The cycle used when a line brings no colour of its own.
CYCLE = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
)

# A quarter of the window stays over the data, so a zoom of one can still be
# dragged and a zoomed window can pass its last sample.
_PAN_KEEP = 0.25
# Twelve pixels on either side is a window. Anything shorter was a click.
_MINIMUM_BAND = 12.0


class _Run:
    __slots__ = ("first", "count", "red", "green", "blue", "alpha", "width", "summarised")

    def __init__(self, first, count, colour, summarised):
        self.first = int(first)
        self.count = int(count)
        self.red, self.green, self.blue = (int(channel) for channel in colour)
        self.alpha = 255
        self.width = 1.0
        self.summarised = bool(summarised)


class _Sample:
    __slots__ = ("line", "x", "y", "px", "py", "red", "green", "blue", "alpha")

    def __init__(self, line, x, y, px, py, colour):
        self.line = int(line)
        self.x = float(x)
        self.y = float(y)
        self.px = float(px)
        self.py = float(py)
        self.red, self.green, self.blue = (int(channel) for channel in colour)
        self.alpha = 255


class _Tick:
    """One mark on the pane. Axis 0 is x, 1 is the common y, 2 is one curve."""

    __slots__ = (
        "x", "y", "value", "axis", "series", "red", "green", "blue", "alpha",
        "labeled", "logarithmic",
    )

    def __init__(self, x, y, value, axis, series, colour, labeled, logarithmic):
        self.x = float(x)
        self.y = float(y)
        self.value = float(value)
        self.axis = int(axis)
        self.series = int(series)
        self.red, self.green, self.blue = (int(channel) for channel in colour)
        self.alpha = 255
        self.labeled = bool(labeled)
        self.logarithmic = bool(logarithmic)


class _Line:
    __slots__ = (
        "y", "x", "explicit_x", "monotonic_x", "colour", "name", "own", "scale_y", "shift_y",
        "y_log", "y_span", "y_low", "y_high", "y_pos", "x_low", "x_high", "x_pos",
        "_cache", "_cached_y",
    )

    def __init__(self, y, colour, name):
        self.y = y
        self.x = None
        self.explicit_x = False
        self.monotonic_x = False
        self.colour = colour
        self.name = name
        self.own = False
        self.scale_y = 1.0
        self.shift_y = 0.0
        self.y_log = False
        self.y_span = 1.0
        self.y_low = self.y_high = self.y_pos = None
        self.x_low = 0.0
        self.x_high = float(y.size) if y.size else 1.0
        self.x_pos = 1.0 if y.size > 1 else None
        self._cache = None
        self._cached_y = None
        self._read_y()

    def column_cache(self):
        """The ladder of this line, or none when it is shorter than a base bucket.

        Kept for this array. A gesture that only moves the window asks the
        same buckets again.
        """
        if self._cached_y is not self.y:
            self._cached_y = self.y
            self._cache = ColumnCache(self.y) if int(self.y.size) >= BASE else None
        return self._cache

    def _read_y(self):
        if self.y.size == 0:
            return
        lo, hi, pos = limits(self.y)
        if lo <= hi:
            self.y_low = float(lo)
            self.y_high = float(hi)
        self.y_pos = float(pos) if math.isfinite(pos) else None

    def set_x(self, x):
        """Index domain when ``x`` is None or does not run one way."""
        count = int(self.y.size)
        if x is None:
            self.x = None
            self.monotonic_x = False
            self.x_low = 0.0
            self.x_high = float(count) if count else 1.0
            self.x_pos = 1.0 if count > 1 else None
            return
        self.x = x
        self.monotonic_x = bool(x.size >= 2 and monotonic(x))
        if not self.monotonic_x:
            # The values are kept, and the fold does not pretend they are a
            # position. The window stays the index, which is the axis the
            # samples actually have.
            self.x_low = 0.0
            self.x_high = float(count) if count else 1.0
            self.x_pos = 1.0 if count > 1 else None
            return
        lo, hi, pos = limits(x)
        self.x_low = float(x[0])
        self.x_high = float(x[-1])
        if lo <= hi:
            self.x_low = float(lo) if not math.isfinite(self.x_low) else self.x_low
            self.x_high = float(hi) if not math.isfinite(self.x_high) else self.x_high
        self.x_pos = float(pos) if math.isfinite(pos) else None


class _Camera:
    """Zoom and pan, in the axis's own scale. Decades while logarithmic."""

    def __init__(self):
        self.zoom_x = 1.0
        self.zoom_y = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.x_log = False
        self.y_log = False
        self.x_base = 10.0
        self.y_base = 10.0
        self.x_min = 0.0
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.x_pos = None
        self.y_pos = None
        self.samples = 0

    def set_extent(self, x_min, x_max, y_min, y_max, x_pos, y_pos, samples):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.x_pos = None if x_pos is None else float(x_pos)
        self.y_pos = None if y_pos is None else float(y_pos)
        self.samples = int(samples)

    def reset(self):
        self.zoom_x = 1.0
        self.zoom_y = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def reset_y(self):
        self.zoom_y = 1.0
        self.pan_y = 0.0

    def max_zoom(self):
        # A handful of samples across the pane. The width of the axis is the
        # wrong length: a short time base of a long record would stop while
        # every column was still a stack of samples.
        if self.samples > 1:
            length = float(self.samples)
        else:
            length = abs(self.x_max - self.x_min) + 1.0
        return max(256.0, length / 16.0)

    def _padded(self, low, high, logarithmic, base):
        if logarithmic:
            if not (low > 0.0) or not (high > 0.0) or not base > 1.0:
                return (1.0, base if base > 1.0 else 10.0)
            from_p = math.log(low) / math.log(base)
            to_p = math.log(high) / math.log(base)
            if not to_p > from_p:
                # A flat line is widened to the powers strictly either side.
                # A value that already is a power takes the ones around it,
                # so 10 on base 10 runs 1..100 and not a zero-width axis.
                at = round(from_p) if abs(from_p - round(from_p)) < 1e-10 else from_p
                exact = abs(at - round(at)) < 1e-10
                from_p = at - 1.0 if exact else math.floor(at)
                to_p = at + 1.0 if exact else math.ceil(at)
            air = (to_p - from_p) * 0.05
            return (base ** (from_p - air), base ** (to_p + air))
        air = (high - low) * 0.05 if high > low else 1.0
        return (low - air, high + air)

    def padded_x(self):
        if self.x_log:
            low = self.x_pos if self.x_pos is not None else self.x_min
            return self._padded(low, self.x_max, True, self.x_base)
        return self._padded(self.x_min, self.x_max, False, self.x_base)

    def padded_y(self):
        if self.y_log:
            low = self.y_pos if self.y_pos is not None else self.y_min
            return self._padded(low, self.y_max, True, self.y_base)
        return self._padded(self.y_min, self.y_max, False, self.y_base)

    def _pos(self, value, logarithmic, base):
        if not logarithmic:
            return float(value)
        if not value > 0.0 or not base > 1.0:
            return 0.0
        return math.log(value) / math.log(base)

    def _value(self, position, logarithmic, base):
        if not logarithmic:
            return float(position)
        return float(base ** position)

    def _clamp_pan(self, pan, zoom, low, high, logarithmic, base):
        full = self._pos(high, logarithmic, base) - self._pos(low, logarithmic, base)
        if not zoom > 0.0 or not full > 0.0:
            return 0.0
        span = full / zoom
        room = full / 2.0 + span * (0.5 - _PAN_KEEP)
        if pan < -room:
            return -room
        if pan > room:
            return room
        return pan

    def _end(self, low, high, zoom, pan, logarithmic, base, sign):
        from_p = self._pos(low, logarithmic, base)
        to_p = self._pos(high, logarithmic, base)
        if not zoom > 0.0:
            zoom = 1.0
        span = (to_p - from_p) / zoom
        centre = (from_p + to_p) / 2.0 + pan
        return self._value(centre + sign * span / 2.0, logarithmic, base)

    def view_min_x(self):
        low, high = self.padded_x()
        return self._end(low, high, self.zoom_x, self.pan_x, self.x_log, self.x_base, -1.0)

    def view_max_x(self):
        low, high = self.padded_x()
        return self._end(low, high, self.zoom_x, self.pan_x, self.x_log, self.x_base, 1.0)

    def view_min_y(self):
        low, high = self.padded_y()
        return self._end(low, high, self.zoom_y, self.pan_y, self.y_log, self.y_base, -1.0)

    def view_max_y(self):
        low, high = self.padded_y()
        return self._end(low, high, self.zoom_y, self.pan_y, self.y_log, self.y_base, 1.0)

    def y_span(self):
        return self._pos(self.view_max_y(), self.y_log, self.y_base) - self._pos(
            self.view_min_y(), self.y_log, self.y_base
        )

    def minimum_span_x(self):
        low, high = self.padded_x()
        return abs(high - low) / self.max_zoom()

    def _log_ceiling(self, full, held, fraction, minimum_span, base):
        def width(span):
            return base ** (held + (1.0 - fraction) * span) - base ** (held - fraction * span)

        if not (full > 0.0) or not (minimum_span > 0.0) or not base > 1.0:
            return self.max_zoom()
        try:
            if not width(full) > minimum_span:
                return 1.0
        except OverflowError:
            return self.max_zoom()
        narrow = 0.0
        wide = full
        for _ in range(60):
            mid = (narrow + wide) / 2.0
            try:
                fits = width(mid) >= minimum_span
            except OverflowError:
                fits = True
            if fits:
                wide = mid
            else:
                narrow = mid
        if not wide > 0.0:
            return self.max_zoom()
        return max(1.0, full / wide)

    def _zoomed(self, zoom, pan, low, high, fraction, factor, logarithmic, base, minimum_span):
        if not (factor > 0.0) or not math.isfinite(factor):
            return zoom, pan
        from_p = self._pos(low, logarithmic, base)
        to_p = self._pos(high, logarithmic, base)
        full = to_p - from_p
        if not full > 0.0 or not zoom > 0.0:
            return zoom, pan
        span = full / zoom
        held = (from_p + to_p) / 2.0 + pan - span / 2.0 + fraction * span
        ceiling = self.max_zoom()
        if logarithmic and minimum_span > 0.0 and base > 1.0:
            ceiling = self._log_ceiling(full, held, fraction, minimum_span, base)
        next_zoom = max(1.0, min(max(ceiling, zoom), zoom * factor))
        next_span = full / next_zoom
        centre = held - fraction * next_span + next_span / 2.0
        next_pan = self._clamp_pan(
            centre - (from_p + to_p) / 2.0, next_zoom, low, high, logarithmic, base
        )
        return next_zoom, next_pan

    def zoom_at(self, fx, fy, factor, only_x, only_y):
        fx = min(1.0, max(0.0, fx))
        fy = min(1.0, max(0.0, fy))
        x_low, x_high = self.padded_x()
        y_low, y_high = self.padded_y()
        if not only_y:
            self.zoom_x, self.pan_x = self._zoomed(
                self.zoom_x, self.pan_x, x_low, x_high, fx, factor, self.x_log, self.x_base,
                self.minimum_span_x(),
            )
        if not only_x:
            self.zoom_y, self.pan_y = self._zoomed(
                self.zoom_y, self.pan_y, y_low, y_high, fy, factor, self.y_log, self.y_base, 0.0,
            )

    def pan_by(self, dx, dy, width, height):
        if not (width > 0.0) or not (height > 0.0):
            return
        x_low, x_high = self.padded_x()
        y_low, y_high = self.padded_y()
        span_x = (
            self._pos(x_high, self.x_log, self.x_base) - self._pos(x_low, self.x_log, self.x_base)
        ) / self.zoom_x
        span_y = (
            self._pos(y_high, self.y_log, self.y_base) - self._pos(y_low, self.y_log, self.y_base)
        ) / self.zoom_y
        self.pan_x = self._clamp_pan(
            self.pan_x - dx * span_x / width, self.zoom_x, x_low, x_high, self.x_log, self.x_base,
        )
        self.pan_y = self._clamp_pan(
            self.pan_y + dy * span_y / height, self.zoom_y, y_low, y_high, self.y_log, self.y_base,
        )

    def _along(self, low, high, at, logarithmic, base):
        from_p = self._pos(low, logarithmic, base)
        to_p = self._pos(high, logarithmic, base)
        return self._value(from_p + at * (to_p - from_p), logarithmic, base)

    def data_x_at(self, px, width):
        if not (width > 0.0):
            return self.view_min_x()
        at = min(1.0, max(0.0, px / width))
        return self._along(self.view_min_x(), self.view_max_x(), at, self.x_log, self.x_base)

    def data_y_at(self, py, height):
        if not (height > 0.0):
            return self.view_min_y()
        # y grows downward on the pane and upward on the axis.
        at = min(1.0, max(0.0, py / height))
        return self._along(self.view_min_y(), self.view_max_y(), 1.0 - at, self.y_log, self.y_base)

    def _viewed(self, low, high, want_from, want_to, logarithmic, base, minimum_span):
        axis_from = self._pos(low, logarithmic, base)
        axis_to = self._pos(high, logarithmic, base)
        want_a = self._pos(want_from, logarithmic, base)
        want_b = self._pos(want_to, logarithmic, base)
        full = axis_to - axis_from
        span = abs(want_b - want_a)
        if not full > 0.0 or not span > 0.0:
            return 1.0, 0.0
        centre = (want_a + want_b) / 2.0
        if logarithmic and minimum_span > 0.0:
            ceiling = self._log_ceiling(full, centre, 0.5, minimum_span, base)
        else:
            ceiling = self.max_zoom()
        zoom = max(1.0, min(ceiling, full / span))
        pan = self._clamp_pan(centre - (axis_from + axis_to) / 2.0, zoom, low, high, logarithmic, base)
        return zoom, pan

    def set_view_range(self, x0, x1, y0, y1):
        x_from, x_to = (min(x0, x1), max(x0, x1))
        y_from, y_to = (min(y0, y1), max(y0, y1))
        x_axis = self.padded_x()
        y_axis = self.padded_y()
        # A bound at or below zero is not a place on a logarithmic axis.
        if self.x_log:
            if not x_to > 0.0:
                x_from, x_to = x_axis
            else:
                x_from = max(x_from, x_axis[0])
        if self.y_log:
            if not y_to > 0.0:
                y_from, y_to = y_axis
            else:
                y_from = max(y_from, y_axis[0])
        self.zoom_x, self.pan_x = self._viewed(
            x_axis[0], x_axis[1], x_from, x_to, self.x_log, self.x_base, self.minimum_span_x(),
        )
        self.zoom_y, self.pan_y = self._viewed(
            y_axis[0], y_axis[1], y_from, y_to, self.y_log, self.y_base, 0.0,
        )

    def zoom_to_region(self, px0, py0, px1, py1, width, height):
        if not (width > 0.0) or not (height > 0.0):
            return False
        px0 = min(width, max(0.0, px0))
        px1 = min(width, max(0.0, px1))
        py0 = min(height, max(0.0, py0))
        py1 = min(height, max(0.0, py1))
        left, right = min(px0, px1), max(px0, px1)
        top, bottom = min(py0, py1), max(py0, py1)
        if right - left < _MINIMUM_BAND or bottom - top < _MINIMUM_BAND:
            return False
        # The top of the band is the larger y. data_y_at already turns the
        # pane upside down, so the lower pixel is the higher value.
        self.set_view_range(
            self.data_x_at(left, width),
            self.data_x_at(right, width),
            self.data_y_at(bottom, height),
            self.data_y_at(top, height),
        )
        return True

    def zoom_x_band(self, px0, px1, width, height):
        """Zoom the time axis to this pixel span. The value axis stays."""
        if not (width > 0.0) or not (height > 0.0):
            return False
        px0 = min(width, max(0.0, px0))
        px1 = min(width, max(0.0, px1))
        left, right = min(px0, px1), max(px0, px1)
        if right - left < _MINIMUM_BAND:
            return False
        x_axis = self.padded_x()
        x_from = self.data_x_at(left, width)
        x_to = self.data_x_at(right, width)
        if self.x_log and not max(x_from, x_to) > 0.0:
            return False
        self.zoom_x, self.pan_x = self._viewed(
            x_axis[0],
            x_axis[1],
            min(x_from, x_to),
            max(x_from, x_to),
            self.x_log,
            self.x_base,
            self.minimum_span_x(),
        )
        return True

    def zoom_y_band(self, py0, py1, width, height):
        """Zoom the value axis to this pixel span. The time axis stays.

        The lower pixel is the larger value, as in ``zoom_to_region``.
        """
        if not (width > 0.0) or not (height > 0.0):
            return False
        py0 = min(height, max(0.0, py0))
        py1 = min(height, max(0.0, py1))
        top, bottom = min(py0, py1), max(py0, py1)
        if bottom - top < _MINIMUM_BAND:
            return False
        y_axis = self.padded_y()
        y_from = self.data_y_at(bottom, height)
        y_to = self.data_y_at(top, height)
        if self.y_log and not max(y_from, y_to) > 0.0:
            return False
        self.zoom_y, self.pan_y = self._viewed(
            y_axis[0],
            y_axis[1],
            min(y_from, y_to),
            max(y_from, y_to),
            self.y_log,
            self.y_base,
            0.0,
        )
        return True

    def line_span(self, scale, shift):
        if not (scale > 0.0) or not math.isfinite(scale):
            scale = 1.0
        low, high = self.padded_y()
        low_p = self._pos(low, self.y_log, self.y_base)
        high_p = self._pos(high, self.y_log, self.y_base)
        from_p = self._pos(self.view_min_y(), self.y_log, self.y_base)
        to_p = self._pos(self.view_max_y(), self.y_log, self.y_base)
        full = high_p - low_p
        view = to_p - from_p
        if not full > 0.0 or not view > 0.0:
            return (
                self._value(from_p + shift, self.y_log, self.y_base),
                self._value(to_p + shift, self.y_log, self.y_base),
            )
        low_f = (from_p - low_p) / full
        high_f = (to_p - low_p) / full
        home = full / scale
        home_from = (low_p + high_p) / 2.0 + shift - home / 2.0
        return (
            self._value(home_from + low_f * home, self.y_log, self.y_base),
            self._value(home_from + high_f * home, self.y_log, self.y_base),
        )

    def place_line(self, low, high, fraction_low, fraction_high):
        if not (fraction_high > fraction_low) or not math.isfinite(low) or not math.isfinite(high):
            return None
        if not high >= low:
            return None
        if self.y_log and (not low > 0.0 or not high > 0.0):
            return None
        band_low, band_high = self._padded(low, high, self.y_log, self.y_base)
        p0 = self._pos(band_low, self.y_log, self.y_base)
        p1 = self._pos(band_high, self.y_log, self.y_base)
        data_span = p1 - p0
        axis_low, axis_high = self.padded_y()
        axis_low_p = self._pos(axis_low, self.y_log, self.y_base)
        axis_high_p = self._pos(axis_high, self.y_log, self.y_base)
        full = axis_high_p - axis_low_p
        from_p = self._pos(self.view_min_y(), self.y_log, self.y_base)
        to_p = self._pos(self.view_max_y(), self.y_log, self.y_base)
        if not full > 0.0:
            return None
        view_frac = (to_p - from_p) / full
        if not data_span > 0.0 or not view_frac > 0.0:
            return None
        vis_span = data_span / (fraction_high - fraction_low)
        vis_from = p0 - fraction_low * vis_span
        low_f = (from_p - axis_low_p) / full
        home = vis_span / view_frac
        if not home > 0.0:
            return None
        scale = full / home
        home_from = vis_from - low_f * home
        shift = home_from + home / 2.0 - (axis_low_p + axis_high_p) / 2.0
        if not (math.isfinite(scale) and math.isfinite(shift) and scale > 0.0):
            return None
        return scale, shift

    def scale_line(self, scale, shift, fraction_up, factor):
        """Zoom one line's y about ``fraction_up``. The value there stays put."""
        if not (factor > 0.0) or not math.isfinite(factor):
            return scale, shift
        if not (scale > 0.0) or not math.isfinite(scale):
            scale = 1.0
        axis_low, axis_high = self.padded_y()
        low_p = self._pos(axis_low, self.y_log, self.y_base)
        high_p = self._pos(axis_high, self.y_log, self.y_base)
        full = high_p - low_p
        from_p = self._pos(self.view_min_y(), self.y_log, self.y_base)
        to_p = self._pos(self.view_max_y(), self.y_log, self.y_base)
        view = to_p - from_p
        if not full > 0.0 or not view > 0.0:
            return scale, shift
        window = self.line_span(scale, shift)
        vis_from = self._pos(window[0], self.y_log, self.y_base)
        vis_to = self._pos(window[1], self.y_log, self.y_base)
        span = vis_to - vis_from
        if not span > 0.0:
            return scale, shift
        held = vis_from + min(1.0, max(0.0, fraction_up)) * span
        min_span = full / self.max_zoom()
        next_span = span / factor
        if factor >= 1.0:
            next_span = max(next_span, min(span, min_span))
        else:
            next_span = min(next_span, max(span, full))
        if not next_span > 0.0:
            return scale, shift
        low_f = (from_p - low_p) / full
        high_f = (to_p - low_p) / full
        view_frac = high_f - low_f
        if not view_frac > 0.0:
            return scale, shift
        new_from = held - min(1.0, max(0.0, fraction_up)) * next_span
        new_home = next_span / view_frac
        if not new_home > 0.0:
            return scale, shift
        scale = full / new_home
        home_from = new_from - low_f * new_home
        shift = home_from + new_home / 2.0 - (low_p + high_p) / 2.0
        return scale, shift


def _chord_at(line, edge, log_x: bool, log_y: bool):
    """Y of the chord at one column edge, or None where the chord is a gap."""
    if line.monotonic_x:
        return _chord_time(line.x, line.y, float(edge), log_x, log_y)
    return y_on_chord(line.y, float(edge), log_x, log_y)


def _append_gap(xs, ys, pxs, pys) -> None:
    xs.append(math.nan)
    ys.append(math.nan)
    pxs.append(math.nan)
    pys.append(math.nan)


def _drawable_sample(x: float, y: float, log_x: bool, log_y: bool) -> bool:
    """A sample the chord may pass through. A NaN or a non-positive log value is a gap."""
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    if log_x and not x > 0.0:
        return False
    if log_y and not y > 0.0:
        return False
    return True


def _sample_window(line, x0: float, x1: float) -> tuple[int, int]:
    """Indices from the sample before the window through the sample after it."""
    count = int(line.y.size)
    if count == 0:
        return 0, 0
    if line.monotonic_x:
        left = int(np.searchsorted(line.x, x0, side="left"))
        right = int(np.searchsorted(line.x, x1, side="right"))
    else:
        left = int(math.ceil(x0 - 1e-9))
        right = int(math.floor(x1 + 1e-9)) + 1
    start = 0 if left <= 0 else min(count, left) - 1
    stop = min(count, max(right, 0) + 1)
    if stop < start:
        stop = start
    return start, stop


def _fraction(value, low, high, logarithmic):
    if logarithmic:
        if not (value > 0.0 and low > 0.0 and high > low):
            return None
        return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    if not high > low:
        return None
    return (value - low) / (high - low)


def _linear_values(low: float, high: float) -> list[float]:
    """About six steps of 1, 2 or 5 times a power of ten, inside the window."""
    span = high - low
    if not math.isfinite(span) or not span > 0.0:
        return []
    raw = span / 6.0
    magnitude = 10.0 ** math.floor(math.log10(raw))
    residual = raw / magnitude
    if residual <= 1.0:
        factor = 1.0
    elif residual <= 2.0:
        factor = 2.0
    elif residual <= 5.0:
        factor = 5.0
    else:
        factor = 10.0
    step = factor * magnitude
    if not step > 0.0:
        return []
    first = math.ceil(low / step - 1e-9)
    last = math.floor(high / step + 1e-9)
    values = []
    for index in range(int(first), int(last) + 1):
        value = index * step
        if low - step * 1e-6 <= value <= high + step * 1e-6:
            values.append(float(value))
        if len(values) >= 24:
            break
    return values


def _inside(value: float, low: float, high: float) -> bool:
    if not value > 0.0:
        return False
    if low <= value <= high:
        return True
    return math.isclose(value, low, rel_tol=1e-8) or math.isclose(value, high, rel_tol=1e-8)


def _log_values(low: float, high: float, base: float) -> tuple[list[float], list[float]]:
    """Powers are labeled. The other integer multiples of a power are marks."""
    labeled: list[float] = []
    plain: list[float] = []
    if not (low > 0.0 and high > low and base > 1.0):
        return labeled, plain
    log_base = math.log(base)
    first = math.floor(math.log(low) / log_base)
    last = math.ceil(math.log(high) / log_base)
    if last - first > 64:
        last = first + 64
    for exponent in range(first, last + 1):
        value = base ** exponent
        if _inside(value, low, high):
            labeled.append(float(value))
    if last - first <= 12:
        multiples = int(math.floor(base + 1e-9))
        for exponent in range(first, last):
            step = base ** exponent
            for factor in range(2, multiples):
                value = factor * step
                if _inside(value, low, high):
                    plain.append(float(value))
    return labeled, plain


def _hdf5_vector(path, dataset: str):
    """The whole 1-D dataset. The file is closed; the array is what is kept."""
    import h5py

    handle = h5py.File(path, "r")
    try:
        data = handle[dataset]
        if getattr(data, "ndim", None) != 1 or not np.issubdtype(data.dtype, np.number):
            raise ValueError(f"{dataset} is not a 1-D numeric dataset")
        array = np.ascontiguousarray(data, dtype=np.float64)
    finally:
        handle.close()
    if array.size < 2:
        raise ValueError(f"{dataset} has fewer than two samples")
    return array


def _segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length = dx * dx + dy * dy
    if length == 0.0:
        return math.hypot(px - ax, py - ay)
    t = min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / length))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class Plot:
    """Lines, the window, and the envelope of whatever that window shows."""

    def __init__(self):
        self._lines: list[_Line] = []
        self._camera = _Camera()
        self._width = 800.0
        self._height = 400.0
        self._ratio = 1.0
        self._pixels = np.zeros((0, 2), dtype=np.float64)
        self._data = np.zeros((0, 2), dtype=np.float64)
        self._runs: list[_Run] = []
        self._owners = np.zeros(0, dtype=np.int32)
        self._segments: list[tuple] = []
        self._vertices: list[tuple] = []
        self._shared_x = None

    def close(self) -> None:
        self.clear()

    def add_line(self, y, colour=None, name=None) -> int:
        if colour is None:
            colour = CYCLE[len(self._lines) % len(CYCLE)]
        colour = (int(colour[0]), int(colour[1]), int(colour[2]))
        text = str(name).strip() if name else ""
        array = np.ascontiguousarray(y, dtype=np.float64)
        line = _Line(array, colour, text or None)
        shared = self._shared_x
        if shared is not None:
            if shared.size < 2 or shared.size != line.y.size:
                raise ValueError("x must have one sample per value of this line, and at least two")
            line.set_x(shared)
        self._lines.append(line)
        self._sync()
        return len(self._lines) - 1

    def set_x(self, x) -> None:
        """One time for every line that has none of its own. None is the index.

        One value per sample of each such line, and at least two. A different
        length is refused, as with ``set_line_x``. A line that already has its
        own time keeps it.
        """
        if x is None:
            self._shared_x = None
            for line in self._lines:
                if not line.explicit_x:
                    line.set_x(None)
            self._sync()
            return
        array = np.ascontiguousarray(x, dtype=np.float64)
        if array.size < 2:
            raise ValueError("x must have one sample per value of this line, and at least two")
        for line in self._lines:
            if line.explicit_x:
                continue
            if array.size != line.y.size:
                raise ValueError("x must have one sample per value of this line, and at least two")
        self._shared_x = array
        for line in self._lines:
            if not line.explicit_x:
                line.set_x(array)
        self._sync()

    def set_x_hdf5(self, path, dataset: str) -> None:
        """Read a 1-D numeric dataset and use it as the shared time.

        The samples are kept. A closer look folds the window it asks for
        and does not read the file again.
        """
        self.set_x(_hdf5_vector(path, dataset))

    def set_line_x_hdf5(self, index: int, path, dataset: str) -> None:
        """Read a 1-D numeric dataset as this line's own time.

        The length has to be this line's, as with ``set_line_x``.
        """
        self.set_line_x(index, _hdf5_vector(path, dataset))

    def add_hdf5(self, path, dataset: str, colour=None, name=None) -> int:
        """Read a 1-D numeric dataset and append it as a line.

        The samples are kept. The fold still sees only the window on screen.
        """
        return self.add_line(_hdf5_vector(path, dataset), colour, name)

    def set_line_x(self, index: int, x) -> None:
        """This line's own time. None puts the line back on the shared time.

        One value per sample, and at least two. A different length is refused:
        index ``i`` of the time is sample ``i``. The array has to go one way
        to be a position; one that doubles back stays an index line. None,
        when no time is shared, is the sample index.
        """
        line = self._line_at(index)
        if x is None:
            line.explicit_x = False
            shared = self._shared_x
            if shared is not None and shared.size == line.y.size:
                line.set_x(shared)
            else:
                line.set_x(None)
            self._sync()
            return
        array = np.ascontiguousarray(x, dtype=np.float64)
        if array.size < 2 or array.size != line.y.size:
            raise ValueError("x must have one sample per value of this line, and at least two")
        line.set_x(array)
        line.explicit_x = True
        self._sync()

    def clear(self) -> None:
        self._lines.clear()
        self._shared_x = None
        self._sync()
        self._pixels = np.zeros((0, 2), dtype=np.float64)
        self._data = np.zeros((0, 2), dtype=np.float64)
        self._runs = []
        self._owners = np.zeros(0, dtype=np.int32)
        self._segments = []
        self._vertices = []

    def line_name(self, index: int):
        if index < 0 or index >= len(self._lines):
            return None
        return self._lines[index].name

    def line_colour(self, index: int):
        if index < 0 or index >= len(self._lines):
            return None
        return self._lines[index].colour

    def named_lines(self):
        """Lines that have a name, in the order they were added, with their colour."""
        return [
            (index, line.name, line.colour)
            for index, line in enumerate(self._lines)
            if line.name
        ]

    def line_count(self) -> int:
        return len(self._lines)

    def set_pane(self, width: int, height: int, pixel_ratio: float = 1.0) -> None:
        self._width = float(width)
        self._height = float(height)
        self._ratio = float(pixel_ratio) if pixel_ratio and pixel_ratio > 0.0 else 1.0

    def set_x_log(self, on: bool) -> None:
        on = bool(on)
        if self._camera.x_log == on:
            return
        self._camera.x_log = on
        # A pan of forty is four decades or it is forty units. The window
        # goes back to the whole line, which is the only reading that is both.
        self._camera.reset()

    def set_y_log(self, on: bool) -> None:
        on = bool(on)
        if self._camera.y_log == on:
            return
        self._camera.y_log = on
        self._camera.reset()

    def set_x_log_base(self, base: float) -> None:
        value = self._base(base)
        if value == self._camera.x_base:
            return
        self._camera.x_base = value
        if self._camera.x_log:
            self._camera.reset()

    def set_y_log_base(self, base: float) -> None:
        value = self._base(base)
        if value == self._camera.y_base:
            return
        self._camera.y_base = value
        if self._camera.y_log:
            self._camera.reset()

    def x_log_base(self) -> float:
        return self._camera.x_base

    def y_log_base(self) -> float:
        return self._camera.y_base

    def x_log(self) -> bool:
        return self._camera.x_log

    def y_log(self) -> bool:
        return self._camera.y_log

    def reset_view(self) -> None:
        self._camera.reset()

    def wheel(self, px: float, py: float, factor: float, shift: bool, control: bool,
              alt: bool = False) -> None:
        """Zoom about a pane-local pixel. A factor above one zooms in.

        Shift alone is x, Ctrl alone is y, and both together is both.
        Alt with Ctrl scales only the curve nearest the pointer, in y,
        and the value under the pointer stays where it is.
        """
        if not (self._width > 0.0) or not (self._height > 0.0):
            return
        fx = float(px) / self._width
        fy = 1.0 - float(py) / self._height
        if alt and control and not shift:
            index = self.nearest(px, py)
            if index < 0:
                return
            line = self._lines[index]
            self._claim(line)
            line.scale_y, line.shift_y = self._camera.scale_line(
                line.scale_y, line.shift_y, fy, float(factor),
            )
            return
        only_x = bool(shift) and not control
        only_y = bool(control) and not shift
        self._camera.zoom_at(fx, fy, float(factor), only_x, only_y)

    def pan(self, dx: float, dy: float) -> None:
        self._camera.pan_by(float(dx), float(dy), self._width, self._height)

    def zoom_rect(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        return self._camera.zoom_to_region(
            float(x0), float(y0), float(x1), float(y1), self._width, self._height,
        )

    def zoom_x_band(self, x0: float, x1: float) -> bool:
        """Zoom x to the pane-local pixel span. Y stays where it is."""
        return self._camera.zoom_x_band(float(x0), float(x1), self._width, self._height)

    def zoom_y_band(self, y0: float, y1: float) -> bool:
        """Zoom y to the pane-local pixel span. X stays where it is."""
        return self._camera.zoom_y_band(float(y0), float(y1), self._width, self._height)

    def set_range(self, x0=None, x1=None, y0=None, y1=None) -> None:
        """Open the window on these values. An axis left out stays as it is.

        On a logarithmic axis a bound at or below zero is not a place, so it
        is clipped to the part of the axis that exists.
        """
        if x0 is None:
            x0 = self.view_min_x()
        if x1 is None:
            x1 = self.view_max_x()
        if y0 is None:
            y0 = self.view_min_y()
        if y1 is None:
            y1 = self.view_max_y()
        self._camera.set_view_range(float(x0), float(x1), float(y0), float(y1))

    def view_min_x(self) -> float:
        return self._camera.view_min_x()

    def view_max_x(self) -> float:
        return self._camera.view_max_x()

    def view_min_y(self) -> float:
        return self._camera.view_min_y()

    def view_max_y(self) -> float:
        return self._camera.view_max_y()

    def data_x_at(self, px: float) -> float:
        return self._camera.data_x_at(float(px), self._width)

    def data_y_at(self, py: float) -> float:
        return self._camera.data_y_at(float(py), self._height)

    def stack_lines(self) -> None:
        """Lay every line in an equal band, the first at the top.

        Each band is that line's own padded span. Y returns to the whole
        axis; x stays. The numbers beside a stacked line are that line's values.
        """
        items = []
        for index, line in enumerate(self._lines):
            if line.y_low is None or line.y_high is None:
                continue
            span = _band_span(self._camera.y_log, line.y_low, line.y_high, line.y_pos)
            if span is None:
                continue
            items.append((index, span[0], span[1]))
        if not items:
            return
        self._camera.reset_y()
        bands = float(len(items))
        for band, (index, low, high) in enumerate(items):
            placed = self._camera.place_line(
                low, high, (bands - 1.0 - band) / bands, (bands - band) / bands,
            )
            if placed is None:
                continue
            line = self._lines[index]
            line.scale_y, line.shift_y = placed
            # place_line used the camera's scale, so the line has to name
            # that one. A scale it had before was in other units.
            line.y_log = self._camera.y_log
            line.own = True

    def set_own_axis(self, index: int, on: bool) -> None:
        line = self._line_at(index)
        if line is None:
            return
        if on:
            self._claim(line)
            return
        line.own = False
        line.shift_y = 0.0
        line.scale_y = 1.0
        line.y_log = False

    def own_axis(self, index: int) -> bool:
        line = self._line_at(index)
        return bool(line and line.own)

    def own_count(self) -> int:
        return sum(1 for line in self._lines if line.own)

    def shared_count(self) -> int:
        return sum(1 for line in self._lines if not line.own)

    def set_line_y_log(self, index: int, on: bool) -> None:
        """Logarithmic y for one line that already has its own axis.

        A line still on the common axis is left alone: that scale is the
        checkbox. Zero and negative samples become gaps, as a NaN does.
        """
        line = self._line_at(index)
        if line is None or not line.own:
            return
        line.y_log = bool(on)

    def line_y_log(self, index: int) -> bool:
        """The scale this line is drawn on. The common axis, when it has none of its own."""
        line = self._line_at(index)
        if line is None:
            return self._camera.y_log
        if not line.own:
            return self._camera.y_log
        return line.y_log

    def line_y_range(self, index: int) -> tuple[float, float]:
        line = self._line_at(index)
        if line is not None and line.own:
            return self._camera.line_span(line.scale_y, line.shift_y)
        return (self.view_min_y(), self.view_max_y())

    def shift_line(self, index: int, dx: float, dy: float) -> None:
        del dx  # x stays on the shared axis; a shift there would rename it.
        line = self._line_at(index)
        if line is None:
            return
        self._claim(line)
        scale = line.scale_y if line.scale_y > 0.0 else 1.0
        if self._height > 0.0 and line.y_span != 0.0:
            line.shift_y += float(dy) / self._height * line.y_span / scale

    def nearest(self, px: float, py: float) -> int:
        best = 14.0
        found = -1
        for line, ax, ay, bx, by in self._segments:
            distance = _segment_distance(float(px), float(py), ax, ay, bx, by)
            if distance <= best:
                best = distance
                found = line
        return found

    def sample(self, px: float, py: float):
        """The drawn sample closest to a pane-local pixel, or None.

        x and y are that sample's values. A shifted line answers in its own y.
        """
        if not (self._width > 0.0) or not (self._height > 0.0):
            return None
        if px < 0.0 or py < 0.0 or px > self._width or py > self._height:
            return None
        best = math.inf
        found = None
        for line, x, y, sx, sy, colour in self._vertices:
            if sx < 0.0 or sy < 0.0 or sx > self._width or sy > self._height:
                continue
            distance = (sx - px) * (sx - px) + (sy - py) * (sy - py)
            # A later line wins a tie: it is the one drawn on top.
            if distance <= best:
                best = distance
                found = _Sample(line, x, y, sx, sy, colour)
        return found

    def project(self):
        """Fold every line into the pane and return pixels, runs and ticks.

        The ticks number the pane the Qt window draws. Matplotlib numbers its
        own axes and ignores them. The points ``data_points`` returns are the
        same vertices in the data's own units. A column with several samples
        is the vertical from its min to its max, and a gap is a NaN so nothing
        is drawn across it. A window where no column holds more than one
        sample is the samples themselves, joined by the chord.
        """
        columns = max(1, int(round(self._width * self._ratio)))
        x0 = self.view_min_x()
        x1 = self.view_max_x()
        log_x = self._camera.x_log
        xs: list[float] = []
        ys: list[float] = []
        pxs: list[float] = []
        pys: list[float] = []
        runs: list[_Run] = []
        owners: list[int] = []
        segments: list[tuple] = []
        vertices: list[tuple] = []
        for index, line in enumerate(self._lines):
            if line.y.size < 2:
                line.y_span = self._camera.y_span()
                continue
            log_y = self.line_y_log(index)
            time = line.x if line.monotonic_x else None
            # Anchored columns: panning slides the window across a fixed grid
            # instead of sliding the grid under the samples. A sample stays in
            # its column, and the line translates instead of boiling.
            mins, maxs, counts, edges = envelope(
                line.y, x0, x1, columns, x=time, log_x=log_x, log_y=log_y,
                cache=line.column_cache(), anchored=True,
            )
            if edges.size != columns + 1:
                line.y_span = self._camera.y_span()
                continue
            low, high = self.line_y_range(index)
            first = len(xs)
            # A column with several samples is a vertical, cut off so it does
            # not run into the next one. A column with one sample or none is
            # the chord, and on a logarithmic x those columns sit beside the
            # crowded ones: breaking them too is the dotted line.
            if int(counts.max()) <= 1:
                self._stroke_samples(
                    xs, ys, pxs, pys, vertices, segments, index, line, x0, x1, low, high,
                    log_x, log_y,
                )
                summarised = False
            else:
                summarised = True
                linked = False
                for column in range(mins.shape[0]):
                    # A crowded column is a vertical only while that vertical
                    # is tall enough to see. On a logarithmic x the first
                    # columns are the narrow ones, and a small amplitude is
                    # then shorter than the pen: an isolated vertical of no
                    # height is a dot, and the gap before the next column is
                    # the dotted line. The chord is the stroke instead.
                    if int(counts[column]) > 1 and self._vertical_shows(
                        float(mins[column]), float(maxs[column]), low, high, log_y,
                    ):
                        if linked:
                            _append_gap(xs, ys, pxs, pys)
                            linked = False
                        lo = float(mins[column])
                        hi = float(maxs[column])
                        mid = 0.5 * (float(edges[column]) + float(edges[column + 1]))
                        self._push(
                            xs, ys, pxs, pys, vertices, index, mid, lo, low, high, log_y,
                            line.colour,
                        )
                        self._push(
                            xs, ys, pxs, pys, vertices, index, mid, hi, low, high, log_y,
                            line.colour,
                        )
                        segments.append((index, pxs[-2], pys[-2], pxs[-1], pys[-1]))
                        _append_gap(xs, ys, pxs, pys)
                        continue
                    y_left = _chord_at(line, edges[column], log_x, log_y)
                    y_right = _chord_at(line, edges[column + 1], log_x, log_y)
                    if y_left is None or y_right is None:
                        if linked:
                            _append_gap(xs, ys, pxs, pys)
                            linked = False
                        continue
                    if not linked:
                        self._push(
                            xs, ys, pxs, pys, vertices, index, float(edges[column]), y_left,
                            low, high, log_y, line.colour,
                        )
                    self._push(
                        xs, ys, pxs, pys, vertices, index, float(edges[column + 1]), y_right,
                        low, high, log_y, line.colour,
                    )
                    segments.append((index, pxs[-2], pys[-2], pxs[-1], pys[-1]))
                    linked = True
            count = len(xs) - first
            if count > 0 and not (xs[-1] == xs[-1]):
                count -= 1
            if count > 0:
                runs.append(_Run(first, count, line.colour, summarised))
                owners.append(index)
            line.y_span = self._camera.y_span()
        self._data = np.column_stack((xs, ys)) if xs else np.zeros((0, 2), dtype=np.float64)
        self._pixels = np.column_stack((pxs, pys)) if pxs else np.zeros((0, 2), dtype=np.float64)
        self._runs = runs
        self._owners = np.asarray(owners, dtype=np.int32)
        self._segments = segments
        self._vertices = vertices
        return self._pixels, runs, self._pane_ticks()

    def _pane_ticks(self) -> list[_Tick]:
        """Marks for the Qt ruler. Axis 2 is one curve's own y, in its colour."""
        ticks: list[_Tick] = []
        if not (self._width > 0.0 and self._height > 0.0):
            return ticks
        ink = (220, 220, 220)
        self._axis_ticks(
            ticks, 0, -1, self.view_min_x(), self.view_max_x(),
            self._camera.x_log, self._camera.x_base, ink, False,
        )
        self._axis_ticks(
            ticks, 1, -1, self.view_min_y(), self.view_max_y(),
            self._camera.y_log, self._camera.y_base, ink, True,
        )
        for index, line in enumerate(self._lines):
            if not line.own:
                continue
            low, high = self.line_y_range(index)
            self._axis_ticks(
                ticks, 2, index, low, high, self.line_y_log(index),
                self._camera.y_base, line.colour, True,
            )
        return ticks

    def _axis_ticks(self, ticks, axis, series, low, high, logarithmic, base, colour, horizontal):
        if logarithmic:
            labeled, plain = _log_values(float(low), float(high), float(base))
            marks = [(value, True) for value in labeled] + [(value, False) for value in plain]
        else:
            marks = [(value, True) for value in _linear_values(float(low), float(high))]
        for value, labeled in marks:
            fraction = _fraction(value, low, high, logarithmic)
            if fraction is None:
                continue
            if horizontal:
                ticks.append(_Tick(
                    0.0, self._height * (1.0 - fraction), value, axis, series, colour, labeled,
                    logarithmic,
                ))
            else:
                ticks.append(_Tick(
                    fraction * self._width, self._height, value, axis, series, colour, labeled,
                    logarithmic,
                ))

    def data_points(self):
        """The vertices of the last project(), in the data's own units."""
        return self._data

    def run_lines(self):
        """Which line each run of the last project() belongs to."""
        return self._owners

    def _vertical_shows(self, lo: float, hi: float, low: float, high: float, log_y: bool) -> bool:
        """True when the column's min and max are at least a pixel apart."""
        if not (lo <= hi) or not math.isfinite(lo) or not math.isfinite(hi):
            return False
        top = _fraction(hi, low, high, log_y)
        bottom = _fraction(lo, low, high, log_y)
        if top is None or bottom is None:
            return False
        return abs(top - bottom) * self._height >= 1.0

    def _stroke_samples(
        self, xs, ys, pxs, pys, vertices, segments, index, line, x0, x1, low, high, log_x, log_y,
    ) -> None:
        """The samples in the window, and the one on either side, joined by the chord."""
        start, stop = _sample_window(line, x0, x1)
        linked = False
        for i in range(start, stop):
            x = float(line.x[i]) if line.monotonic_x else float(i)
            y = float(line.y[i])
            if not _drawable_sample(x, y, log_x, log_y):
                if linked:
                    xs.append(math.nan)
                    ys.append(math.nan)
                    pxs.append(math.nan)
                    pys.append(math.nan)
                    linked = False
                continue
            self._push(xs, ys, pxs, pys, vertices, index, x, y, low, high, log_y, line.colour)
            if linked:
                segments.append((index, pxs[-2], pys[-2], pxs[-1], pys[-1]))
            linked = True

    def _push(self, xs, ys, pxs, pys, vertices, index, x, y, low, high, log_y, colour):
        xs.append(x)
        ys.append(y)
        fx = _fraction(x, self.view_min_x(), self.view_max_x(), self._camera.x_log)
        fy = _fraction(y, low, high, log_y)
        px = fx * self._width if fx is not None else math.nan
        py = self._height * (1.0 - fy) if fy is not None else math.nan
        pxs.append(px)
        pys.append(py)
        if fx is not None and fy is not None:
            vertices.append((index, x, y, px, py, colour))

    def _sync(self):
        x_lo = x_hi = y_lo = y_hi = None
        x_pos = y_pos = None
        samples = 0
        for line in self._lines:
            samples = max(samples, int(line.y.size))
            if x_lo is None:
                x_lo, x_hi = line.x_low, line.x_high
            else:
                x_lo = min(x_lo, line.x_low)
                x_hi = max(x_hi, line.x_high)
            if line.x_pos is not None:
                x_pos = line.x_pos if x_pos is None else min(x_pos, line.x_pos)
            if line.y_low is None or line.y_high is None:
                continue
            if y_lo is None:
                y_lo, y_hi = line.y_low, line.y_high
            else:
                y_lo = min(y_lo, line.y_low)
                y_hi = max(y_hi, line.y_high)
            if line.y_pos is not None:
                y_pos = line.y_pos if y_pos is None else min(y_pos, line.y_pos)
        if x_lo is None:
            x_lo, x_hi = 0.0, 1.0
        if y_lo is None:
            y_lo, y_hi = 0.0, 1.0
        if not x_hi > x_lo:
            x_hi = x_lo + 1.0
        if not y_hi > y_lo:
            y_lo -= 0.5
            y_hi += 0.5
        self._camera.set_extent(x_lo, x_hi, y_lo, y_hi, x_pos, y_pos, samples)

    def _line_at(self, index: int):
        index = int(index)
        if index < 0 or index >= len(self._lines):
            return None
        return self._lines[index]

    def _claim(self, line: _Line) -> None:
        if not line.own:
            line.y_log = self._camera.y_log
        line.own = True

    @staticmethod
    def _base(base: float) -> float:
        value = float(base)
        if not math.isfinite(value) or not value > 1.0:
            raise ValueError("a logarithmic base is a number above one")
        return value


def _band_span(logarithmic, low, high, positive):
    if not math.isfinite(low) or not math.isfinite(high) or not high >= low:
        return None
    if logarithmic:
        if not high > 0.0:
            return None
        if not low > 0.0:
            if positive is None or not positive > 0.0 or not high > positive:
                return None
            low = positive
    return (low, high)
