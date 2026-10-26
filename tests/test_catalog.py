# SPDX-FileCopyrightText: 2026 Andreas Martin
# SPDX-License-Identifier: BSD-2-Clause

"""A ``.npy`` is listed from its header and drawn with the sample index as time."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from linefold.catalog import Catalog
from linefold.plot import Plot


def _save(path: Path, array) -> Path:
    np.save(path, array)
    return path


def _open(path: Path) -> Catalog:
    catalog = Catalog()
    entry = catalog.add(path)
    assert entry.loaded is None
    for slot in entry.channels:
        slot.checked = True
    catalog.load(entry)
    return catalog


def test_a_vector_is_one_channel(tmp_path):
    path = _save(tmp_path / "walk.npy", np.array([1.0, 2.0, 4.0]))
    catalog = _open(path)
    entry = catalog.files[0]
    assert entry.time_unit == ""
    assert entry.dt == 1.0
    assert [(slot.name, slot.samples, slot.unit) for slot in entry.channels] == [
        ("walk", 3, "")
    ]
    plot = Plot()
    assert catalog.publish(plot) == ""
    line = plot._lines[0]
    assert np.array_equal(line.y, [1.0, 2.0, 4.0])
    assert np.array_equal(line.x, [0.0, 1.0, 2.0])


def test_columns_are_channels_when_the_matrix_is_long(tmp_path):
    values = np.arange(12, dtype=np.int32).reshape(4, 3)
    catalog = _open(_save(tmp_path / "long.npy", np.asfortranarray(values)))
    entry = catalog.files[0]
    assert [slot.name for slot in entry.channels] == ["0", "1", "2"]
    assert all(slot.samples == 4 for slot in entry.channels)
    plot = Plot()
    catalog.publish(plot)
    assert np.array_equal(plot._lines[1].y, [1, 4, 7, 10])


def test_rows_are_channels_when_the_matrix_is_short(tmp_path):
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    catalog = _open(_save(tmp_path / "short.npy", values))
    assert [slot.samples for slot in catalog.files[0].channels] == [4, 4, 4]
    plot = Plot()
    catalog.publish(plot)
    assert np.array_equal(plot._lines[1].y, [4, 5, 6, 7])


def test_add_does_not_read_the_samples(tmp_path, monkeypatch):
    path = _save(tmp_path / "header.npy", np.arange(8, dtype=np.float64))

    def refuse(*_args, **_kwargs):
        raise AssertionError("add read the samples")

    monkeypatch.setattr(np, "load", refuse)
    entry = Catalog().add(path)
    assert entry.loaded is None
    assert entry.channels[0].samples == 8


def test_a_second_import_of_the_same_file_is_refused(tmp_path):
    # NumPy only treats a lower-case suffix as already present, so the
    # upper-case name is the file the viewer is asked for.
    stored = _save(tmp_path / "walk.npy", np.array([0.0, 1.0]))
    path = stored.with_name("Walk.NPY")
    stored.rename(path)
    catalog = Catalog()
    catalog.add(path)
    try:
        catalog.add(path)
    except FileExistsError:
        return
    raise AssertionError("the same path was imported twice")


def test_a_text_array_is_refused(tmp_path):
    path = _save(tmp_path / "names.npy", np.array(["a", "b"]))
    try:
        Catalog().add(path)
    except ValueError as exc:
        assert "not a numeric array" in str(exc)
        return
    raise AssertionError("a text array was imported")


def test_three_axes_are_refused(tmp_path):
    path = _save(tmp_path / "cube.npy", np.zeros((2, 2, 2)))
    try:
        Catalog().add(path)
    except ValueError as exc:
        assert "shape (2, 2, 2)" in str(exc)
        return
    raise AssertionError("a cube was imported")


def test_one_sample_is_not_a_curve(tmp_path):
    path = _save(tmp_path / "dot.npy", np.array([1.0]))
    try:
        Catalog().add(path)
    except ValueError as exc:
        assert "at least two samples" in str(exc)
        return
    raise AssertionError("a single sample was imported")
