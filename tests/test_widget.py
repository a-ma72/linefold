# SPDX-FileCopyrightText: 2026 Andreas Martin
# SPDX-License-Identifier: GPL-3.0-only

"""A NaN in a run is a gap, so the stroke does not join the columns through it."""

from __future__ import annotations

import math

import numpy as np

from linefold.widget import _finite_spans, _pixel_centres


def test_a_nan_splits_the_run():
    points = np.array(
        [[0.0, 1.0], [0.0, 2.0], [np.nan, np.nan], [1.0, 3.0], [1.0, 4.0]],
        dtype=np.float64,
    )
    assert _finite_spans(points) == [(0, 2), (3, 5)]


def test_a_sample_stays_on_its_pixel():
    # Anything in one device pixel is drawn at that pixel's centre, so a pan
    # across the pixel does not move the stroke.
    x = np.array([10.0, 10.2, 10.9, 11.0, np.nan])
    centres = _pixel_centres(x, 1.5)
    assert np.allclose(centres[:3], np.floor(x[:3] * 1.5) + 0.5)
    assert centres[3] == math.floor(11.0 * 1.5) + 0.5
    assert math.isnan(centres[4])
    assert _pixel_centres(np.array([10.01]), 1.0)[0] == _pixel_centres(np.array([10.99]), 1.0)[0]


def test_a_run_of_only_gaps_is_not_stroked():
    points = np.array([[np.nan, 0.0], [1.0, np.nan]], dtype=np.float64)
    assert _finite_spans(points) == []
