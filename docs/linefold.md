# linefold

linefold keeps a line in memory and folds whatever window is on screen into one column per device pixel. The fold is the picture. `PlotWidget` and `MplView` only stroke what `envelope` already reduced.

## The column envelope

A column is a vertical slot of the pane. Over the open interval `[x0, x1)` the window is split into as many columns as the pane has device pixels. Each column stores three numbers: the smallest finite sample that landed in it, the largest, and how many samples that was.

Sample `i` of a line with no time of its own sits at `x = i`. A monotonic time base is the same split read the other way: the edges are times, and a sample belongs to the column whose edges contain its time.

Two hosts draw this. Neither decides the shape of the line.

## Folding and fast display

Folding here is the column envelope. It is not a convolution. Nothing is weighted, and a sample is not smeared into its neighbours. A column reports the extremes of the samples that fall inside it, then the chord corrects the two edges.

### One column per pixel

`Plot.project` asks for `max(1, round(width * pixel_ratio))` columns between `view_min_x` and `view_max_x`. The edges are anchored. A column is still one device pixel wide, but the grid is locked to zero (to one, on a logarithmic x axis) and phased so that the sample under a pixel's centre stays in that pixel. A grid locked to the window would slide every boundary by a fraction of a sample as the pane moves, and the line would boil while it was panned. Anchoring makes the line translate instead.

On a linear axis the edges are equally spaced. On a logarithmic x axis they are equally spaced in the logarithm. The base cancels, so any base above one produces the same edges. A logarithmic window that is not strictly positive has no edges.

Sample `i` belongs to the column that contains `x = i`. A sample that sits on an edge is kept in the column on the right.

### What a column keeps

The scan walks each column once. A NaN fails both comparisons, so it is not an extreme and it does not add to the count. On a logarithmic y axis a value at or below zero is left out in the same way: the positive-only scan is a second pass over the same samples, not a filter applied afterwards.

The chord is applied after that scan. The y on an edge is the straight line between the sample before the edge and the sample after it, evaluated where the edge actually sits. Half is only the case where the edge is in the middle. That y is offered to both columns that share the edge, and it lengthens a column when it lies outside the samples already inside. A NaN on either side of the edge, or a non-positive value on a logarithmic axis, means the chord is not drawn and does not move the extremes.

`envelope` returns four arrays: the minimum of each column, the maximum, the count, and the edges (`columns + 1` of them).

### What is stroked

`project` turns those arrays into vertices.

When no column holds more than one sample, the window is close enough to see the samples. The stroke is the samples themselves, joined by the chord. That is the line, not a summary.

When a column holds several samples, and the vertical from its minimum to its maximum is at least one pixel tall, the stroke is that vertical, placed at the middle of the column. A gap is written after it. The gap is a NaN in the polyline, so the pen does not join one column to the next through empty space. A column with one sample, or none, stays on the chord.

A crowded column whose vertical is shorter than a pixel is not drawn as a dot. On a logarithmic x axis the first columns are the narrow ones, and a small amplitude there is shorter than the pen. An isolated vertical of no height, followed by a gap, would read as a dotted line. The chord is the stroke instead.

A gap in the data stays a gap. Logarithmic axes drop values they cannot draw, and those dropped values are gaps too.

### The ladder

A line of at least `BASE` samples (1024) keeps a ladder of the same fold, built once in `ColumnCache`. A shorter line has no ladder. Every later window of a long line reads the ladder instead of walking the samples again.

The bottom rung cuts the line into buckets of 1024 samples. The last bucket is shorter when the line does not fill it, and it still starts on the alignment. Each bucket stores six values, twice:

- the first sample and the last sample
- the minimum, the maximum, and the count of finite samples
- the same three numbers counted only where the sample is strictly positive

The positive copy is what a logarithmic y axis reads. Building it once means a log view does not rescan the line.

Coarser rungs are exact merges of the rung below. A group of children becomes one bucket. The minimum and maximum are the minimum and maximum of the children that hold a sample. An empty child does not move the extremes. The count is the sum of the counts. The first sample of the bucket is the first child's first sample, and the last sample is the last child's last sample. The tail group is short and still exact.

The fan, how many children become one parent, is chosen once for the ladder:

- fours all the way up
- twos all the way up
- a taper: fours, then threes, then twos

The taper is the one that is kept when, over a fixed set of spans a long line actually asks for, it reads no more buckets than the twos and stores no more buckets than the fours. The small rungs dominate the store, so the extra large sizes the taper keeps are the whole of the difference. When that test fails, the ladder is built in fours.

A span does not walk every rung. It asks for the coarsest bucket that starts where the cursor is and still ends inside the span. Samples between alignments are not buckets. They are one slice of the raw line, read up to the next multiple of 1024. The chord at a column edge is still the two raw samples beside that edge, so the picture is the one the plain scan draws.

### The fast path, and the same answer without it

The rungs are packed into flat arrays, coarse first. A window of a long line is hundreds of columns, and pulling buckets out of Python objects was the whole cost. With numba installed (`pip install "linefold[numba]"`), one parallel pass reads those arrays. Without numba, the three answers are computed in NumPy.

NumPy does not imitate the parallel walk. A column narrower than `BASE` is a plain reduction of its samples. A wider column uses the base buckets when those buckets hold more samples than the two fringes around them. A fringe is at most two base widths, one at each end, and gathering it costs more per sample than one reduction straight through the column. The unaligned edges are gathered and reduced once. Fewer buckets than that, and the plain reduction is the smaller read. A NaN still fails both comparisons, and a value at or below zero is still left out of a logarithmic column. The two paths agree.

The ladder is not rebuilt when the window moves. `project` passes `line.column_cache()` into `envelope` on every frame.

### Drawing the reduced line

`project` returns pixels, runs, and ticks. A run is a slice of the pixel array, the colour, and a flag that says whether the run is a summary of columns or the samples themselves. The ticks number the pane the Qt window draws. Matplotlib numbers its own axes and ignores them. `data_points` is the same vertices in the data's own units.

The hosts do not fold. They stroke.

`PlotWidget` paints the runs onto a plain image in device pixels, with no transform on the painter, and copies that image into the pane. An envelope of a whole line is thousands of segments. Qt's raster engine draws that in a few milliseconds on a plain image, and in hundreds of milliseconds when the painter carries a scale. A device-pixel ratio on the image installs such a scale, so the ratio is applied to the coordinates instead, and the image is the size of the pane in device pixels.

A summary is a vertical per column, a hairpin. A pen wider than one device pixel turns that hairpin into the slow stroke, so a summary is stroked one pixel wide. Antialiasing is off when the run is dense or a summary. A NaN inside one polyline is not a gap for Qt's stroker: the stroker fills the pane. Each run is split into finite spans, and each span is its own polyline. A run that is not a summary, and that has few samples on the pane, also marks those samples. A summary stays a summary when a logarithmic axis drops values. Fewer vertices is not a closer look.

`MplView` draws the same vertices in the data's own units. A gap stays a gap, so matplotlib does not stitch across it. The axes are matplotlib's: its ticks, its logarithm, its limits taken from the camera. The navigation toolbar stays off. A scroll, a drag, and a rectangle go through `Plot`. Left to itself, matplotlib would zoom whatever was last drawn, and that is a pane of the line, not the line.

## Plot

`Plot` holds the lines, the camera, and the last projection.

```python
plot = Plot()
index = plot.add_line(y, colour=(0, 114, 178), name="channel")
plot.set_line_x(index, time)
plot.set_pane(width, height, pixel_ratio)
plot.project()
```

`add_line` copies `y` to a contiguous `float64` array. Without a colour, the line takes the next entry of `CYCLE`. The return value is the line's index.

`set_x` is one time for every line that does not have its own. `None` means the sample index. The array needs one value per sample of each such line, and at least two. A line that already has its own time keeps it. `set_line_x` sets that private time. A different length is refused.

`set_pane` is the pixel size of the drawing area. The column count follows the width and the pixel ratio.

The camera keeps a quarter of the window over the data, so a zoom of one can still be dragged and a zoomed window can pass its last sample. `wheel` zooms at a pixel. Shift zooms x only, Control zooms y only. `pan` moves the window. `zoom_rect` opens the rectangle dragged on the pane. `reset_view` puts the window back over the data. `set_range` sets it explicitly. `view_min_x`, `view_max_x`, `view_min_y`, and `view_max_y` are the window. `data_x_at` and `data_y_at` convert a pixel back to data.

`set_x_log` and `set_y_log` switch the shared axes. `set_x_log_base` and `set_y_log_base` choose the base. The base numbers the powers. It does not move the column edges.

A line can leave the shared y axis. `set_own_axis` gives it a private scale. `stack_lines` lays every line in an equal band, the first at the top, and the numbers beside a stacked line are that line's values. `set_line_y_log` is logarithmic y for a line that already has its own axis. A line still on the common axis is left alone. `shift_line` moves a line on its own y axis. X stays shared. A shift there would rename the axis.

`nearest` and `sample` answer a pixel from the vertices of the last `project`. A later line wins a tie, because it is the one drawn on top.

`clear` drops the lines. `close` is `clear`.

## Hosts

### PlotWidget

`PlotWidget` is a PySide6 widget. It is GPL-3.0-only, and importing it needs the `qt` extra:

```bash
pip install "linefold[qt]"
```

Construct it with an optional parent. `add_line`, `set_x`, `set_line_x`, `set_x_log`, `set_y_log`, the log bases, `set_range`, and `stack_lines` forward to the `Plot` it owns. `set_x_label` and `set_y_label` name the axes. `show_lines` replaces the checked set.

The widget calls `project` again when the pane is resized and when the camera moves. The wheel, the buttons, and the modifiers are the gestures in [docs/demos.md](demos.md#mouse-and-wheel). A readout follows the pointer. The y names of private axes sit in their own column.

Missing PySide6 or shiboken6 raises `ImportError` and names `pip install linefold[qt]`.

### MplView

`MplView` is the matplotlib host. Importing `linefold.mpl` needs the `mpl` extra:

```bash
pip install "linefold[mpl]"
```

```python
from linefold.mpl import MplView

view = MplView()
view.plot.add_line(y, name="channel")
view.set_x_label("s")
view.set_y_label("m/s²")
view.redraw()
```

Pass an existing `Plot` or a matplotlib `figure` to the constructor. `show` opens a window with the toolbar hidden. `redraw` projects and replaces the artists. Private y axes are matplotlib twin axes, seated outward from the shared spine, in the line's colour. The y tick labels and the y name are smaller than the x axis.

The same gestures as the Qt widget go through the camera. They are listed in [docs/demos.md](demos.md#mouse-and-wheel). Shift and Control on the rectangle itself are the Qt viewer only. Here the right button always opens both axes, and a click that is too short to be a window still opens **Logarithmic** on a private y axis.

## Files

### Catalog

`Catalog` lists RPC3 files and `.npy` arrays for the two viewers. It does not know about Qt or Tk. Both windows call it and draw the result themselves.

`add` reads the header and appends the file. The samples stay unread. A path already in the catalog raises `FileExistsError`. `FileNotFoundError`, `rpc3.FileFormatError`, and `ValueError` propagate, and the caller leaves that file out.

An RPC3 file is read with [`rpc3-file`](https://pypi.org/project/rpc3-file/), which is installed with linefold. `read` looks `progressbar` up in the module that defines it. That name lives on `rpc3.rpc3` and is not re-exported, so the catalog turns the bar off there. Otherwise tqdm would write into a window that has no console. Channels of one RPC3 file share `DELTA_T` and start at zero. Files do not share a clock, so each drawn channel carries its own time.

A `.npy` is one numeric vector, or a matrix of channels. The header is enough for the tree. The samples are read the first time a channel of that file is shown, and kept. A vector uses the file name. Several columns are numbered, because a `.npy` has no channel names. A matrix with at least as many rows as columns is samples by channels. Fewer rows than columns is channels by samples. A square stays samples by channels. A `.npy` has no clock, so its time is the sample index and its time unit is empty.

`load` reads every channel of one file. A second call keeps the first read. `publish` replaces the plot's lines with the checked channels and returns the y unit when every drawn channel shares one, otherwise `""`. A channel with fewer than two samples is not a curve. The view is left where it was.

### HDF5

`add_hdf5`, `set_x_hdf5`, and `set_line_x_hdf5` read one 1-D numeric dataset and keep the array. A closer look folds the window it asks for and does not read the file again. The dataset has to be one-dimensional and numeric, with at least two samples. Importing h5py is deferred to the call. Install it with:

```bash
pip install "linefold[hdf5]"
```

## Examples

The two viewers, the file tree, and the mouse gestures are described in [docs/demos.md](demos.md).

```bash
python examples/demo.py
python examples/demo_mpl.py
```

`examples/make_samples.py` writes the RPC3 and NumPy files those windows can import. They land in `examples/samples`. The checkout does not ship them, and neither the script nor that folder is installed with the package.

## Licences

| Path | Licence |
| --- | --- |
| `src/linefold/widget.py` | GPL-3.0-only |
| `examples/demo.py` | GPL-3.0-only |
| `tests/test_widget.py` | GPL-3.0-only |
| every other file in this package | BSD-2-Clause |

The texts are [LICENSE](../LICENSE) and [LICENSES/GPL-3.0-only.txt](../LICENSES/GPL-3.0-only.txt). Each source file names its licence in an SPDX header. BSD-2-Clause is compatible with GPL-3.0, so the files may sit in one tree. A BSD file used on its own stays BSD. A distribution that includes the three GPL files has to convey GPL-3.0 with them, including this licence text.
