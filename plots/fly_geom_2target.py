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

'''
Two-target fly geometry: the self-consistent-equilibrium BRANCH diagram, alone.

Companion to ``fly_geom.py``'s ``fly_geom_3target_branch.png``: the same branch
panel (panel B there -- every SC equilibrium heading along the y=0 cut vs
observer x, stable filled blue / unstable open red, with the bifurcation
x-locations marked and labelled) rendered as its own single-panel 300-dpi png
``fly_geom_2target_branch.png``, for the TWO-target fly instead of the three.

Only the target geometry changes. The model parameterization, the scan window
and the panel rendering are *imported* from ``fly_geom`` rather than copied, so
the two figures stay comparable by construction:

  * warp/weight/K -- ``fly_geom.build_model`` (foveal ``lin_cutoff`` warp with
    the weight tied to it, ``A_WARP``/``B_WARP``, K=4.5). Illustration knobs,
    not the empirical fit -- hence the same "not fit to data" side-note.
  * beta -- ``fly_geom.beta_for`` reproduces the earlier per-target temperature
    T=0.2 as N_targets/T, so this two-target scene gets beta = 2/0.2 = 10 where
    its three-target sibling gets 15. (Same reason ``decision_skeleton.FLY2``
    carries beta=20 against ``FLY``'s 30 under that module's T=0.10.)
  * the y=0 cut and the x-window (-1, 4) -- ``fly_geom.BRANCH_Y0`` /
    ``BRANCH_XLIM``, kept identical so the two panels read against each other
    directly. The targets sit at x=4.33, just past the right edge.

Geometry: GODM fly2 -- 60 deg separation at distance 5, i.e. two circle targets
of radius 0.5 at +-30 deg -> (4.330, +-2.5). Matches ``two_target_fly_refine.py``
and ``decision_skeleton.FLY2``.

Reading the figure: with no on-midline target the two-target case is a plain **Y**
-- one pitchfork at x~0.20 where the straight-ahead compromise heading goes
unstable and the two committed arms are born, then the two interior unstable
branches folding back onto the midline at x~1.23. There is no second bifurcation
cascade (that is the three-target centre-target route). The unstable midline
branch then simply STOPS at x~2.33: its coherence has decayed to R<0.01, the
threshold at which ``sc_equilib`` discards an incoherent solution
(``neural_band_model.py`` line 631) -- the observer is drawing level with the two
targets, which pull to opposite sides. That truncation is the filter, not a
bifurcation, so it carries no dashed line.

Run:  python fly_geom_2target.py
      FLYGEOM_FAST=1 python fly_geom_2target.py   # coarse/fast layout check
'''

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'walker_analysis'), HERE):
    sys.path.insert(0, p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model setup, scan window and panel renderer all come from the 3-target figure
# script -- single source of truth, so the two branch panels differ only in the
# target geometry. (Importing it also picks up its Computer-Modern mathtext and
# font-size rcParams, and its FLYGEOM_FAST resolution switch.)
from fly_geom import (build_model, draw_branch_panel, mixed_suptitle,
                      BRANCH_XLIM, BRANCH_Y0, DPI)

OUT_2TARGET = os.path.join(HERE, 'fly_geom_2target_branch.png')

# GODM fly2 geometry: 60 deg separation at distance 5 -> +-30 deg, (4.330, +-2.5).
_HALF = np.pi / 6.0                       # half the 60 deg separation
TARGET_LOCS = np.array([[5.0*np.cos(_HALF),  5.0*np.sin(_HALF)],
                        [5.0*np.cos(_HALF), -5.0*np.sin(_HALF)]])


def figure_2target():
    '''The y=0 SC-equilibrium branch diagram for the two-target fly.'''
    neur_model = build_model(TARGET_LOCS)

    # Single panel, sized to match the height of the 3-target figure so the two
    # can be set side by side at the same scale.
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    bif_x = draw_branch_panel(ax, neur_model, y0=BRANCH_Y0, xlim=BRANCH_XLIM)
    print('  y=0 bifurcations at x ~ '
          + (', '.join(f'{b:.2f}' for b in bif_x) if bif_x.size else '(none)'))

    mixed_suptitle(fig, 'Two-Target Fly Geometry',
                   '(for illustration: not fit to data)')

    fig.savefig(OUT_2TARGET, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT_2TARGET)


if __name__ == '__main__':
    figure_2target()
