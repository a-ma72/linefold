# SPDX-FileCopyrightText: 2026 Andreas Martin
# SPDX-License-Identifier: BSD-2-Clause

"""Write a few RPC3 and NumPy files the viewers can import.

    python examples/make_samples.py

Files land in ``examples/samples``. Drop them on ``examples/demo.py`` or
``examples/demo_mpl.py``.

``ride.rpc`` is 16-bit, three channels, three units, one millisecond.
``burst.rsp`` is 32-bit float, two channels, one unit, ten milliseconds,
so the two files do not share a clock. ``trace.npy`` is one long vector.
``wheel.npy`` is five million samples of wheel force from road excitation.
``columns.npy`` is samples by channels. ``rows.npy`` has fewer rows than
columns, so the catalog reads it as channels by samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "samples"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    _quiet_rpc3()
    _ride()
    _burst()
    _trace()
    _wheel()
    _columns()
    _rows()


def _ride() -> None:
    """Three channels, mixed units, integer RPC3."""
    import rpc3

    dt = 1e-3
    t = _time(65536, dt)
    rng = np.random.default_rng(1)
    speed = 50.0 + 15.0 * np.sin(2 * np.pi * 0.2 * t) + rng.normal(0, 0.4, t.size)
    lateral = 3.0 * np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    steer = 25.0 * np.sin(2 * np.pi * 0.15 * t + 0.4)
    channels = [
        rpc3.Channel("speed", "km/h", dt, speed),
        rpc3.Channel("ay", "m/s2", dt, lateral),
        rpc3.Channel("steer", "deg", dt, steer),
    ]
    _write_rpc3("ride.rpc", channels, datatype=int)


def _burst() -> None:
    """Two channels, one unit, a slower float clock than ``ride.rpc``."""
    import rpc3

    dt = 1e-2
    t = _time(8192, dt)
    decay = np.exp(-t / 8.0)
    shock = 12.0 * decay * np.sin(2 * np.pi * 4.0 * t)
    residual = 2.0 * decay * np.cos(2 * np.pi * 9.0 * t)
    channels = [
        rpc3.Channel("shock", "m/s2", dt, shock),
        rpc3.Channel("ring", "m/s2", dt, residual),
    ]
    _write_rpc3("burst.rsp", channels, datatype=float)


def _trace() -> None:
    """One numeric vector. The channel name is the file stem."""
    n = 131072
    x = np.linspace(0.0, 80.0, n)
    y = np.sin(x) + 0.25 * np.sin(7.0 * x)
    y[n // 3 : n // 3 + 40] += 4.0
    _save("trace.npy", y.astype(np.float64))


def _wheel() -> None:
    """Wheel force from road excitation, five million samples.

    Coloured noise is the roughness, and a few stretches are rougher or
    smoother. The pulses are small bumps: a short extra load, then a
    smaller unload. They stay well below a kerb strike.
    """
    n = 5_000_000
    rng = np.random.default_rng(7)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freq = np.fft.rfftfreq(n, d=1.0)
    f = np.maximum(freq, freq[1])
    peak = 1.0 / 120.0
    shape = (f / peak) * np.exp(1.0 - f / peak)
    shape[0] = 0.0
    spectrum *= shape
    dynamic = np.fft.irfft(spectrum, n=n)
    dynamic *= 220.0 / np.std(dynamic)

    for start, stop, gain in (
        (0.15, 0.28, 0.55),
        (0.28, 0.46, 1.8),
        (0.62, 0.74, 1.4),
        (0.86, 0.95, 0.65),
    ):
        i0, i1 = int(start * n), int(stop * n)
        dynamic[i0:i1] *= _fade(i1 - i0, gain)

    _bumps(dynamic, rng)
    _save("wheel.npy", 4500.0 + dynamic)


def _fade(length: int, gain: float) -> np.ndarray:
    """Ramp a roughness section in and out so the level does not step."""
    fade = np.ones(length)
    edge = min(8_000, length // 2)
    t = np.linspace(0.0, np.pi / 2.0, edge)
    fade[:edge] = np.sin(t) ** 2
    fade[-edge:] = np.cos(t) ** 2
    return 1.0 + (gain - 1.0) * fade


def _bumps(dynamic: np.ndarray, rng: np.random.Generator) -> None:
    """Isolated load pulses with a short unload behind each one."""
    width = 36
    k = np.arange(width, dtype=np.float64)
    pulse = np.exp(-((k - 4.0) ** 2) / 8.0) - 0.35 * np.exp(-((k - 16.0) ** 2) / 50.0)
    pulse /= np.max(np.abs(pulse))

    gap = 28_000
    count = 120
    slots = (dynamic.size - width) // gap
    chosen = rng.choice(slots, size=count, replace=False)
    starts = chosen * gap + rng.integers(0, gap // 3, size=count)
    amps = rng.uniform(1000.0, 1700.0, size=count)
    for start, amp in zip(starts, amps):
        dynamic[start : start + width] += amp * pulse


def _columns() -> None:
    """Samples by channels: at least as many rows as columns."""
    n = 12000
    t = np.linspace(0.0, 6.0, n)
    columns = np.column_stack(
        (
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * 0.5 * t),
            0.3 * t,
        )
    )
    _save("columns.npy", columns.astype(np.float64))


def _rows() -> None:
    """Fewer rows than columns: the catalog treats each row as a channel."""
    n = 20000
    t = np.linspace(0.0, 4.0, n)
    rows = np.vstack(
        (
            np.sin(2 * np.pi * 2.0 * t),
            np.sign(np.sin(2 * np.pi * 0.25 * t)),
        )
    )
    _save("rows.npy", rows.astype(np.float64))


def _time(count: int, dt: float) -> np.ndarray:
    return np.arange(count, dtype=np.float64) * dt


def _quiet_rpc3() -> None:
    import rpc3.rpc3 as impl

    impl.progressbar = False


def _write_rpc3(name: str, channels: list, **kwargs) -> None:
    import rpc3

    path = OUT / name
    rpc3.write(str(path), channels, overwrite=True, **kwargs)
    print(path)


def _save(name: str, array: np.ndarray) -> None:
    path = OUT / name
    np.save(path, array)
    print(path)


if __name__ == "__main__":
    main()
