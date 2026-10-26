# SPDX-FileCopyrightText: 2026 Andreas Martin
# SPDX-License-Identifier: BSD-2-Clause

"""RPC3 and NumPy files for the two viewer windows.

A file is multiplexed, so one channel is not a read of its own. The header
is enough for the tree. The samples are read once, the first time a channel
of that file is shown, and kept. Both windows call this and draw the result
themselves: nothing here knows about Qt or Tk.

Each drawn channel carries its own time. Channels of one RPC3 file share
``DELTA_T`` and start at zero, files do not, and one shared x would lay a
shorter or differently clocked file on the wrong samples. A ``.npy`` has no
clock of its own, so its time is the sample index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .plot import CYCLE


@dataclass
class ChannelSlot:
    """One channel as the tree shows it, before or after its samples exist."""

    index: int
    name: str
    unit: str
    samples: int
    colour: tuple[int, int, int]
    checked: bool = False


@dataclass
class Rpc3File:
    path: Path
    dt: float
    channels: list[ChannelSlot]
    loaded: list | None = None
    # ``s`` for an RPC3 ``DELTA_T``. Empty for a ``.npy``, whose step is the
    # sample index and must not be labelled as seconds.
    time_unit: str = "s"


@dataclass
class _Samples:
    """One numeric column, with the ``data`` and ``time`` ``publish`` reads."""

    data: object
    dt: float

    def time(self):
        import numpy as np

        count = int(np.asarray(self.data).shape[0])
        return np.arange(count, dtype=np.float64) * self.dt


@dataclass
class Catalog:
    """The files the viewer has imported, in the order the tree shows them."""

    files: list[Rpc3File] = field(default_factory=list)
    _keys: set[str] = field(default_factory=set)
    _colour: int = 0

    def contains(self, path: Path) -> bool:
        try:
            key = _identity(path)
        except OSError:
            return False
        return key in self._keys

    def add(self, path: Path) -> Rpc3File:
        """Read the header and append the file. The samples stay unread.

        A path already in the catalog is refused. ``FileNotFoundError``,
        ``rpc3.FileFormatError`` and ``ValueError`` propagate: the caller
        leaves that file out. A ``.npy`` is one numeric vector, or a matrix
        of channels; anything else is a ``ValueError``.
        """
        path = path.resolve()
        key = _identity(path)
        if key in self._keys:
            raise FileExistsError(path)
        if path.suffix.lower() == ".npy":
            entry = self._entry_npy(path)
        else:
            entry = self._entry_rpc3(path)
        self.files.append(entry)
        self._keys.add(key)
        return entry

    def _entry_rpc3(self, path: Path) -> Rpc3File:
        rpc3 = _rpc3()
        channels, params = rpc3.read(str(path), header_only=True)
        if not channels:
            raise rpc3.FileFormatError(f"{path.name} has no channels")
        samples = params.get("SAMPLES")
        if samples is None:
            samples = int(params["FRAMES"]) * int(params["PTS_PER_FRAME"])
        slots = [
            self._slot(index, str(channel.name), str(channel.unit), int(samples))
            for index, channel in enumerate(channels)
        ]
        return Rpc3File(path=path, dt=float(params["DELTA_T"]), channels=slots)

    def _entry_npy(self, path: Path) -> Rpc3File:
        samples, count = _npy_channels(path)
        # One column keeps the file name. Several are numbered, because a
        # ``.npy`` has no channel names of its own.
        names = [path.stem] if count == 1 else [str(index) for index in range(count)]
        slots = [self._slot(index, name, "", samples) for index, name in enumerate(names)]
        return Rpc3File(path=path, dt=1.0, channels=slots, time_unit="")

    def _slot(self, index: int, name: str, unit: str, samples: int) -> ChannelSlot:
        colour = CYCLE[self._colour % len(CYCLE)]
        self._colour += 1
        return ChannelSlot(index=index, name=name, unit=unit, samples=samples, colour=colour)

    def load(self, entry: Rpc3File) -> None:
        """Read every channel of one file. A second call keeps the first read."""
        if entry.loaded is not None:
            return
        if entry.path.suffix.lower() == ".npy":
            _load_npy(entry)
            return
        rpc3 = _rpc3()
        channels, _params = rpc3.read(str(entry.path))
        if not isinstance(channels, list) or len(channels) != len(entry.channels):
            raise rpc3.FileFormatError(f"{entry.path.name} changed while it was open")
        entry.loaded = channels
        for slot, channel in zip(entry.channels, channels):
            slot.samples = int(len(channel.data))

    def pending(self) -> Rpc3File | None:
        """The first file that is checked somewhere and not read yet."""
        for entry in self.files:
            if entry.loaded is None and any(slot.checked for slot in entry.channels):
                return entry
        return None

    def remove(self, entry: Rpc3File) -> None:
        if entry in self.files:
            self.files.remove(entry)
        self._keys.discard(_identity(entry.path))
        entry.loaded = None

    def publish(self, plot) -> str:
        """Replace the plot's lines with the checked channels.

        Returns the y unit when every drawn channel shares one, otherwise "".
        A channel with fewer than two samples is not a curve. The view is
        left where it was: the window resets once, after this returns.
        """
        drawn = []
        units: list[str] = []
        for entry in self.files:
            if entry.loaded is None:
                continue
            stem = entry.path.stem
            for slot in entry.channels:
                if not slot.checked:
                    continue
                channel = entry.loaded[slot.index]
                data = channel.data
                if data is None or len(data) < 2:
                    continue
                drawn.append((data, channel.time(), slot.colour, f"{stem}: {slot.name}"))
                units.append(slot.unit)
        plot.clear()
        for data, time, colour, name in drawn:
            index = plot.add_line(data, colour=colour, name=name)
            plot.set_line_x(index, time)
        if not units or any(unit != units[0] for unit in units):
            return ""
        return units[0]


def _identity(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _npy_header(path: Path):
    """Shape and dtype from the ``.npy`` preamble. The samples stay on disk."""
    import numpy as np

    with path.open("rb") as handle:
        major, _minor = np.lib.format.read_magic(handle)
        # 2.0 and 3.0 share the 2.0 header reader. 3.0 only widened the
        # field-name encoding; a numeric array's shape is unchanged.
        read = (
            np.lib.format.read_array_header_1_0
            if major == 1
            else np.lib.format.read_array_header_2_0
        )
        shape, _fortran, dtype = read(handle)
    return tuple(int(axis) for axis in shape), np.dtype(dtype)


def _npy_channels(path: Path) -> tuple[int, int]:
    """``(samples, channels)`` for one numeric vector or matrix.

    A matrix with at least as many rows as columns is samples × channels,
    the same first axis a vector already uses. Fewer rows than columns is
    channels × samples: a handful of long recordings stored the other way.
    A square stays samples × channels.
    """
    shape, dtype = _npy_header(path)
    if dtype.kind not in "biuf":
        raise ValueError(f"{path.name} is not a numeric array ({dtype})")
    if len(shape) == 1:
        samples, count = shape[0], 1
    elif len(shape) == 2:
        rows, cols = shape
        samples, count = (rows, cols) if rows >= cols else (cols, rows)
    else:
        raise ValueError(
            f"{path.name} must be a vector or a matrix of channels, not shape {shape}"
        )
    if count < 1 or samples < 2:
        raise ValueError(f"{path.name} needs at least two samples")
    return samples, count


def _load_npy(entry: Rpc3File) -> None:
    import numpy as np

    array = np.load(entry.path, allow_pickle=False)
    samples, count = _npy_channels(entry.path)
    if count != len(entry.channels) or samples < 2:
        raise ValueError(f"{entry.path.name} changed while it was open")
    if array.ndim == 1:
        columns = [array]
    elif array.shape[0] >= array.shape[1]:
        columns = [array[:, index] for index in range(array.shape[1])]
    else:
        columns = [array[index] for index in range(array.shape[0])]
    entry.loaded = [
        _Samples(np.ascontiguousarray(column, dtype=np.float64), entry.dt)
        for column in columns
    ]
    for slot, column in zip(entry.channels, entry.loaded):
        slot.samples = int(column.data.shape[0])


def _rpc3():
    """The installed ``rpc3-file`` package, with its progress bar off.

    ``read`` looks ``progressbar`` up in the module that defines it. The
    installed package keeps that name in ``rpc3.rpc3`` and does not re-export
    it, so setting it on ``rpc3`` would leave the bar on. tqdm would otherwise
    write into a window that has no console.
    """
    try:
        import rpc3
        import rpc3.rpc3 as impl
    except ImportError as error:
        raise ImportError(
            "rpc3-file is not installed; pip install rpc3-file"
        ) from error
    impl.progressbar = False
    return rpc3
