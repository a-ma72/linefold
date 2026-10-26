"""Install linefold from the src layout."""

from pathlib import Path
import re

from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
init = (root / "src" / "linefold" / "__init__.py").read_text(encoding="utf-8")
version = re.search(r'^__version__ = ["\']([^"\']+)["\']', init, re.M).group(1)
readme = (root / "README.md").read_text(encoding="utf-8")

setup(
    name="linefold",
    version=version,
    description="Column envelope for long lines, drawn by Qt or matplotlib.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Andreas Martin",
    license="BSD-2-Clause AND GPL-3.0-only",
    license_files=["LICENSE", "LICENSES/GPL-3.0-only.txt"],
    url="https://github.com/a-ma72/plotlib",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "rpc3-file>=1.0.0rc6",
    ],
    extras_require={
        "qt": ["PySide6", "shiboken6"],
        "mpl": ["matplotlib"],
        "hdf5": ["h5py"],
        "numba": ["numba"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: BSD License",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
