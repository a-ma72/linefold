# SPDX-FileCopyrightText: 2026 Andreas Martin
# SPDX-License-Identifier: BSD-2-Clause

"""Clear errors when an optional extra is not installed."""

from __future__ import annotations

import importlib
from typing import Any


def require(module: str, extra: str) -> Any:
    """Import ``module`` or raise ImportError naming the pip extra.

    Parameters
    ----------
    module : str
        Top-level module to import, for example ``"PySide6"``.
    extra : str
        Extra name used in ``pip install linefold[extra]``.

    Returns
    -------
    Any
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if "pip install linefold[" in str(exc):
            raise
        missing = exc.name or module
        msg = (
            f"linefold needs {missing} for this feature. "
            f"Install it with: pip install linefold[{extra}]"
        )
        raise ImportError(msg) from exc
