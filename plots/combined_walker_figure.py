# Neural Ring Model: Ising-type dynamics of spatial decision-making.
# Copyright (C) 2026 Christopher Strickland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Montage the six finished publication panels into one labelled figure.

Composites the existing 300-dpi panel PNGs (no downscaling -- native pixels are
pasted, so the result is no worse than 300 dpi) into a 2x3 grid:

    row 1 (A B C):  locust-2 skeleton | fly-2 skeleton | fly 2-target walkers
    row 2 (D E F):  locust-3 skeleton | fly-3 skeleton | fly 3-target walkers

Each panel's baked-in title is cropped off and replaced with the short title that
follows that panel in the reordering; a bold panel letter (A-F) is overlaid in the
heatmap corner. Column spacing is uniform (no divider line). Output: a single PNG
tagged 300 dpi.

Run:  python plots/combined_walker_figure.py
"""
import os

import numpy as np
import matplotlib
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

# (letter, filename, replacement title) in reading order: row0 = A B C, row1 = D E F.
PANELS = [
    ('A', 'skeleton_locust2.png',   'Locust stable-track skeleton'),
    ('B', 'skeleton_fly2.png',      'Fly stable-track skeleton'),
    ('C', 'fly_results_2target.png','Fly 2-target random walk'),
    ('D', 'skeleton_locust.png',    'Locust stable-track skeleton'),
    ('E', 'skeleton_fly.png',       'Fly stable-track skeleton'),
    ('F', 'fly_results_3target.png','Fly 3-target random walk'),
]
NCOLS = 3
OUT_BASE = 'combined_walker_figure'

# --- layout (pixels at native 300-dpi panel resolution) ---
MARGIN     = 30      # outer white border
COL_GAP    = 60      # uniform horizontal gap between columns (matches col1-col2)
ROW_GAP    = 50      # vertical gap between the two rows
TITLE_PAD  = 120     # vertical strip above each panel for its title
TITLE_FS   = 56      # title font size (px)
LETTER_FS  = 96      # panel-letter font size (px)
LETTER_INSET = 34    # letter offset from the heatmap top-left corner

_FONT_DIR = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
TITLE_FONT  = ImageFont.truetype(os.path.join(_FONT_DIR, 'DejaVuSans.ttf'), TITLE_FS)
LETTER_FONT = ImageFont.truetype(os.path.join(_FONT_DIR, 'DejaVuSans-Bold.ttf'), LETTER_FS)


def _load_rgb(path):
    """Open as RGB, compositing any alpha over white."""
    im = Image.open(path)
    if im.mode == 'RGBA':
        bg = Image.new('RGB', im.size, 'white')
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert('RGB')


def _crop_title_and_margins(im, white=240, pad=4):
    """Remove the baked-in title, then trim the surrounding white margins. The plot
    band (heatmap + axis tick labels) is by far the longest run of content rows, so
    we crop off everything above it -- that is exactly the title -- and keep the rest
    (axis labels below/left are interior content, not trimmed)."""
    arr = np.asarray(im)
    gray = arr.mean(axis=2)
    content = (gray < white).sum(axis=1) > 3          # a row with real ink
    # longest run of consecutive content rows == the plot band; its start is the crop
    best_len, best_start, i, n = 0, 0, 0, len(content)
    while i < n:
        if content[i]:
            j = i
            while j < n and content[j]:
                j += 1
            if (j - i) > best_len:
                best_len, best_start = j - i, i
            i = j
        else:
            i += 1
    arr = arr[best_start:]
    # trim white margins on all four sides (axis labels are kept -- only outer white)
    gray = arr.mean(axis=2)
    rmask = (gray < white).any(axis=1)
    cmask = (gray < white).any(axis=0)
    r0, r1 = np.argmax(rmask), len(rmask) - np.argmax(rmask[::-1])
    c0, c1 = np.argmax(cmask), len(cmask) - np.argmax(cmask[::-1])
    arr = arr[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad]
    return Image.fromarray(arr)


def _heatmap_corner(im, sat_thresh=0.20):
    """Top-left corner of the saturated (viridis heatmap) region, so the panel letter
    lands on the dark heatmap rather than the white axis margin."""
    arr = np.asarray(im).astype(float)
    mx, mn = arr.max(2), arr.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    ys, xs = np.where(sat > sat_thresh)
    if ys.size == 0:
        return LETTER_INSET, LETTER_INSET
    return int(xs.min()), int(ys.min())


def main():
    panels = []
    for letter, fn, title in PANELS:
        im = _crop_title_and_margins(_load_rgb(os.path.join(HERE, fn)))
        panels.append((letter, title, im))
        print('%s  %-26s cropped -> %d x %d' % (letter, fn, im.width, im.height))

    nrows = (len(panels) + NCOLS - 1) // NCOLS
    grid = [panels[r * NCOLS:(r + 1) * NCOLS] for r in range(nrows)]

    col_w = [max(grid[r][c][2].width for r in range(nrows)) for c in range(NCOLS)]
    row_h = [max(p[2].height for p in grid[r]) for r in range(nrows)]

    x_col = [MARGIN + sum(col_w[:c]) + c * COL_GAP for c in range(NCOLS)]
    canvas_w = MARGIN + sum(col_w) + (NCOLS - 1) * COL_GAP + MARGIN
    y_title = [MARGIN + r * (TITLE_PAD + ROW_GAP) + sum(row_h[:r]) for r in range(nrows)]
    y_panel = [yt + TITLE_PAD for yt in y_title]
    canvas_h = y_panel[-1] + row_h[-1] + MARGIN

    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)

    for r in range(nrows):
        for c, (letter, title, im) in enumerate(grid[r]):
            px = x_col[c] + (col_w[c] - im.width) // 2      # centre in the column box
            py = y_panel[r]                                  # top-align in the row band
            canvas.paste(im, (px, py))
            # title centred over the panel, in the strip above it
            draw.text((px + im.width / 2, y_title[r] + TITLE_PAD / 2), title,
                      font=TITLE_FONT, fill='black', anchor='mm')
            # panel letter on the heatmap corner (white with a black outline -> legible)
            hx, hy = _heatmap_corner(im)
            draw.text((px + hx + LETTER_INSET, py + hy + LETTER_INSET), letter,
                      font=LETTER_FONT, fill='white', anchor='la',
                      stroke_width=4, stroke_fill='black')

    png = os.path.join(HERE, OUT_BASE + '.png')
    canvas.save(png, dpi=(300, 300))
    print('canvas %d x %d px (%.2f x %.2f in @ 300 dpi)'
          % (canvas_w, canvas_h, canvas_w / 300.0, canvas_h / 300.0))
    print('wrote', png)


if __name__ == '__main__':
    main()
