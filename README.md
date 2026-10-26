<p align="center">
  <img src="docs/linefold-logo.png" alt="linefold" width="632">
</p>

# linefold

linefold draws a long line by folding it into one column per pixel. Each column keeps the smallest and the largest sample that lands in it. Qt and matplotlib both stroke that envelope. Neither walks the samples again.

Python 3.11 or newer is required.

## Install

```bash
pip install linefold
```

`numpy` and [`rpc3-file`](https://pypi.org/project/rpc3-file/) are installed with the package. `rpc3-file` is still a pre-release; the dependency asks for `1.0.0rc6` or newer so pip will take it.

Optional extras:

```bash
pip install "linefold[qt]"      # PySide6 widget
pip install "linefold[mpl]"     # matplotlib host
pip install "linefold[hdf5]"    # 1-D datasets as a line or as time
pip install "linefold[numba]"   # faster column scans
```

## A line

```python
import numpy as np
from linefold import Plot

y = np.sin(np.linspace(0, 40, 200_000))
plot = Plot()
plot.add_line(y, name="sine")
plot.set_pane(800, 400)
pixels, runs, ticks = plot.project()
```

`project` folds the window into 800 columns and returns the pixels a host would stroke. A window, the wheel, and the file tree live in `examples/demo.py` (Qt) and `examples/demo_mpl.py` (matplotlib inside Tk). How those viewers behave is described in [docs/demos.md](docs/demos.md).

The fold, the ladder that answers a wide column without rereading the line, and the way the two hosts draw that result are described in [docs/linefold.md](docs/linefold.md).

## Licence

`widget.py`, `examples/demo.py`, and `tests/test_widget.py` are GPL-3.0-only. The rest of the package is BSD-2-Clause. The texts are [LICENSE](LICENSE) and [LICENSES/GPL-3.0-only.txt](LICENSES/GPL-3.0-only.txt). A file keeps the licence named in its SPDX header. The BSD modules can be used on their own under BSD-2-Clause. A distribution that includes the GPL files has to honour GPL-3.0 for those files.
