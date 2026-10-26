# The viewers

`examples/demo.py` and `examples/demo_mpl.py` are the same viewer twice. A tree lists files. Checked channels are drawn on one plot. The Qt window uses `PlotWidget`. The Tk window uses `MplView`, so matplotlib strokes the same envelope and numbers the axes itself.

Neither script is installed with the package. From a checkout, with linefold installed:

```bash
python examples/demo.py
python examples/demo_mpl.py
```

The checkout does not ship recordings. `examples/make_samples.py` writes a few RPC3 and NumPy files into `examples/samples` so the viewers have something to open:

```bash
python examples/make_samples.py
```

Drop that folder on either window. The script is not installed with the package.

| | Qt | matplotlib |
| --- | --- | --- |
| Script | `examples/demo.py` | `examples/demo_mpl.py` |
| Licence | GPL-3.0-only | BSD-2-Clause |
| Install | `pip install "linefold[qt]"` | `pip install "linefold[mpl]"` |
| Dropping files | built in | needs `tkinterdnd2` |
| Log x, log y, base, stack | buttons under the pane | not in this window; the pane gestures still scale one curve |

Both windows are titled **RPC3 Viewer** and open at 1100 by 640. The file list is on the left, about 280 pixels wide. The plot takes the rest.

## What a file becomes

**Import…** opens a dialog. The filters are RPC3 (`*.rpc`, `*.rsp`, `*.tim`, `*.drv`), NumPy (`*.npy`), and every file. Several files can be chosen at once. A file can also be dropped on the window.

Only a path that is a file is considered. A path already in the catalog is skipped. Anything else is handed to `Catalog.add`, which reads the header and leaves the samples on disk. `FileNotFoundError`, `rpc3.FileFormatError`, `ValueError`, and any other error from that read are collected and shown together in one warning, one line per file. If the dialog returned paths and none of them was a file, the status line says **No files.**

The tree shows one row per file, expanded, and one row per channel under it. An RPC3 channel is its name and, when it has one, its unit in brackets. A `.npy` vector is named after the file. Several columns are numbered, because a `.npy` has no channel names of its own. In the Qt window the file row's tooltip is the full path, and a channel's tooltip is the sample count. After the samples have been read, that count is refreshed. An RPC3 tooltip also shows `dt` and the unit `s`.

Channels of one RPC3 file share `DELTA_T` and start at zero. Files do not share a clock, so each drawn channel carries its own time. A `.npy` has no clock. Its time is the sample index. Both viewers still label the x axis `s`. The y axis is labelled only when every drawn channel has the same unit. Mixed units leave it blank. A channel with fewer than two samples is not a curve. The line's name on the plot is the file stem, a colon, and the channel name.

Checking a channel does not read the file yet if another channel of that file was already shown. The samples are read once, the first time any channel of the file is checked, and kept. A second check uses the copy. An RPC3 file is multiplexed, so that read is the whole file. It runs off the window's thread. The status line says **Reading** and the file name while it runs. Only one file is read at a time. If several checked files are still unread, the next starts when the current one finishes.

The picture is updated before that read returns, from whatever is already loaded. When the read fails, every channel of that file is unchecked and a warning names the file and the error. Closing the Qt window disconnects the finished signal so a late read does not draw into a window that is gone. The matplotlib window polls a queue every 50 milliseconds and only then touches Tk. A call into Tk from the reader is refused unless the main loop is already running.

Removing a file drops it from the catalog and from the tree, and the plot is published again. The right button does this. In the Qt tree the menu is **Remove** on the file, including when the click was on one of its channels. In the Tk tree the same menu selects the file row first.

## The Qt window

`examples/demo.py` needs PySide6 and shiboken6. The script asks for them through `linefold._require` and tells you `pip install linefold[qt]` when they are missing. The file is GPL-3.0-only because it is the Qt viewer.

The tree header is **Files**. Each row has a checkbox. Checking the file checks every channel. Unchecking it clears them. A file with some channels checked and some not is partially checked. Checking one channel updates that mark. The change is applied on the next turn of the event loop, so a click that rewrites the children does not start a second update for each of them.

Under the plot:

- **Log y** and **Log x** switch the shared axes.
- **base 10**, **base 2**, and **base e** are one button. Each click advances it and sets both logarithmic bases. The base numbers the powers. It does not move the column edges.
- **Stack** lays every line in an equal band, the first at the top. Each band is that line's own padded span. The numbers beside a stacked line are that line's values.

A line under those buttons is the short form of the gestures: the left button pans, the right button pulls a rectangle, Shift and Control on the wheel zoom one axis, and Alt or the middle button gives the curve under the pointer its own axis.

The status line starts as **Import files or drop them here.** It clears once a file is in the tree and nothing is being read. Dropping works on the window, the tree, the plot, and the bar of buttons. A drop lands on the child under the pointer, so each of those widgets accepts the drop itself. A parent that accepted drops would not see a drop the child ignored.

Every time the checked set changes, `show_lines` replaces the lines and then resets the camera. The window comes back over the data. A pan or a zoom from before the click is not kept.

## The matplotlib window

`examples/demo_mpl.py` needs matplotlib. Tk is in the standard library. The matplotlib toolbar stays off. The navigation is the camera, not matplotlib's zoom of whatever was last drawn.

The tree has no checkboxes. A mark in the row says the state: ☐ none, ☑ checked, ▣ some channels of the file. A click on a channel toggles it. A click on the file checks every channel, or clears them when they were all checked. A click on the disclosure triangle only opens or closes the row.

There is no **Log y**, **Log x**, **base**, or **Stack** button. Logarithmic scales and a private axis still come from the pane, as below. Stacking every line into bands is the Qt button only.

**Import...** is three ASCII dots. The ellipsis is not in the core X font this window uses, and Tk would paint its UTF-8 bytes.

Dropping files needs `tkinterdnd2`. Without it the window still opens, the dialog still imports, and the status line says **Dropping files needs tkinterdnd2 (pip install tkinterdnd2).** With it, a drop is accepted on the window, the tree, and the canvas. The canvas is the matplotlib widget packed into the right pane.

The right button on the plot always opens both axes. Shift and Control during that drag are the Qt viewer only. A click too short to be a window still opens **Logarithmic** on a private y axis.

## Mouse and wheel

Both panes send the pointer through the same camera. The place under the pointer is the place that stays put.

The wheel zooms about the pointer, on x and y together. One notch is a factor of 1.25. Rolling up zooms in. Shift held down zooms x only. Control held down zooms y only. Shift and Control together zoom both, as the plain wheel does. Control and Alt together scale only the curve nearest the pointer, and only in y. That curve takes its own axis, and the value under the pointer stays where it is.

The left button pans. A quarter of the window stays over the data, so the line cannot be dragged entirely off the pane. A left double-click puts the window back over the data. A curve that already has its own axis keeps that axis, including its offset and its scale.

The middle button, or the left button with Alt, drags the curve under the pointer up and down. The drag gives it a private y axis if it did not have one. X stays shared. Control and the left button, on a curve that already has a private axis, put it back on the shared axis. The offset and the private scale go with it, because the shared numbers would otherwise name a place the curve has left.

The right button pulls a rectangle. The corners and the width are labelled in the units of the axes while the button is down. On release that rectangle becomes the window. A drag shorter than twelve pixels on either side is a click, and a click does not zoom. In the Qt viewer, Shift during the drag zooms x only and leaves y, and Control zooms y only and leaves x. A Shift drag is also refused when the time span is that short, and a Control drag when the value span is. In the matplotlib viewer the right button always opens both axes.

A right click on the column of a private y axis opens **Logarithmic**. That toggles a logarithmic y scale for that curve. Zero and negative samples become gaps. A right click in the pane opens nothing.

With no button down, the pointer reads the sample that was drawn, on the axis that drew it. The hair sits on that sample, not on the value under the pointer if the two differ. A shifted curve answers in its own y. A later line wins a tie, because it is the one drawn on top. Leaving the pane clears the reading. In the Qt window a clipped axis name shows its full text as a tooltip while the pointer is on it.
