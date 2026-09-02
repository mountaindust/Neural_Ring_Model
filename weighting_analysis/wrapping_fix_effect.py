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

"""Measure what the 2026-08-04 wrapping-extent fix did to the "ears".

`_get_target_signals` used to pass the CLOSEST target's raw angular extent to
`_integrate_neural_weight`. An extent straddling +-pi comes back from
`get_percep_angles` as a wrapping pair (lo > hi), which integrates to a NEGATIVE
arc length, which the `G > 0` visibility filter then silently discarded -- so
the nearest target vanished from perception for the whole angular window in
which it straddled the rear branch cut.

This script reverts *exactly* that one line and nothing else, then recounts the
upper ear both ways. The trick is that the fixed call site uses `self`::

    intervals = self._unwrap_interval(original_extents[n])     # the fix

while `_subtract_intervals_circle` (which was always correct) calls the same
helper on the class::

    PerceptionModel._unwrap_interval(iv)

so overriding `_unwrap_interval` in a **subclass** reverts the fix at the one
call site that had the bug and leaves the blocking arithmetic untouched.

Result (see README.md, "The wrapping-extent fix, and why it mattered here"):
the bug erased ~31% of the ear, entirely on the UNIFORM side, in a band at the
target's own x -- the notch that split the old ear mask into two lobes.

Run:
    python weighting_analysis/wrapping_fix_effect.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multiprocessing import Pool

import decision_model as dm
from parallel_config import get_n_workers

# Same scene as ears_figure.py. NOTE: the 2026-05 run of this comparison used
# the 'coupled' criterion, which has since been REMOVED (it linearized an
# incomplete equation -- see NeuralBandModel._discrim_reduced). This now runs
# under 'reduced', so the absolute counts are not directly comparable to that
# write-up's numbers; the wrapping-fix effect it measures is unaffected.
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])
TARGET_R = 0.5
WARP, A_WARP, B_WARP = 'cutoff', 0.0, np.pi
CRITERION = 'reduced'
# BETA is the neural Boltzmann factor. This scene has 2 targets and the
# write-up's run used the earlier per-target temperature T=0.2, whose effective
# coupling was N_targets/T, so beta = 2/0.2 = 10 reproduces those numbers.
K, BETA = 2.0, 10.0

# The upper ear only (the lower is its mirror), at a resolution fine enough to
# resolve the notch without a long run.
XS = np.linspace(3.0, 6.0, 31)
YS = np.linspace(1.2, 3.0, 19)


class BuggyPerceptionModel(dm.PerceptionModel):
    """PerceptionModel with the pre-fix (non-unwrapping) behaviour restored."""

    @staticmethod
    def _unwrap_interval(interval):
        return [tuple(interval)]


def build(tied, buggy):
    targets = dm.Targets(locs=TARGET_LOCS, geom_name='circle', r=TARGET_R)
    cls = BuggyPerceptionModel if buggy else dm.PerceptionModel
    pm = cls(targets, (0, 0), 0, neural_angle_dist=WARP,
             angle_weight='neural_angle_dist' if tied else None,
             a_warp=A_WARP, b_warp=B_WARP)
    return dm.NeuralBandModel(pm, beta=BETA, K=K)


_MODEL = None


def _init(cfg):
    global _MODEL
    _MODEL = build(*cfg)


def _count(pt):
    _ang, stab = _MODEL.sc_equilib(focal_loc=pt, stability_criterion=CRITERION)
    return int(np.sum(stab))


def worked_example(loc=(4.5, 2.0)):
    """Print the single-cell arithmetic behind the whole effect."""
    o = np.array(loc, dtype=float)
    targets = dm.Targets(locs=TARGET_LOCS, geom_name='circle', r=TARGET_R)
    dist = np.linalg.norm(TARGET_LOCS - o, axis=1)
    bear = np.arctan2(TARGET_LOCS[:, 1] - o[1], TARGET_LOCS[:, 0] - o[0])
    far = int(np.argmax(dist))
    near = 1 - far
    heading = bear[far]              # candidate: commit to the FAR target
    ext = targets.get_percep_angles(o, heading)
    lo, hi = (float(v) for v in ext[near])
    width = (hi - lo) if hi >= lo else (hi - lo + 2 * np.pi)

    print(f'\n=== worked example at observer {tuple(o)} ===')
    print(f'  candidate heading = the FAR target ({np.degrees(heading):+.1f} deg allo)')
    print(f'  near target: d={dist[near]:.2f}, ego='
          f'{np.degrees(dm.convert_angles(bear[near] - heading)):+.1f} deg, '
          f'visual extent {np.degrees(width):.1f} deg')
    print(f'  extent straddles +-pi: {lo > hi}')
    for tied, lab in ((True, 'FULL   '), (False, 'UNIFORM')):
        pm = build(tied, False).percep_model
        raw = pm._integrate_neural_weight([(lo, hi)])
        fixed = pm._integrate_neural_weight(pm._unwrap_interval((lo, hi)))
        flo, fhi = (float(v) for v in ext[far])
        far_G = pm._integrate_neural_weight(pm._unwrap_interval((flo, fhi)))
        print(f'  {lab}: raw(buggy)={raw:+.3f} -> discarded;  '
              f'correct={fixed:+.3f}, far={far_G:+.3f}  '
              f'=> correct rho(near)={fixed/(fixed+far_G):.3f}, buggy=0.000')


def main():
    n_workers = get_n_workers()
    print(f'workers: {n_workers}   criterion: {CRITERION}')
    worked_example()

    pts = [(float(x), float(y)) for y in YS for x in XS]
    counts = {}
    for buggy in (False, True):
        for tied in (True, False):
            with Pool(n_workers, initializer=_init,
                      initargs=((tied, buggy),)) as pool:
                counts[(buggy, tied)] = np.array(pool.map(_count, pts)).reshape(
                    len(YS), len(XS))
            print(f'  counted buggy={buggy} tied={tied}', flush=True)

    cell = (XS[1] - XS[0]) * (YS[1] - YS[0])
    print(f'\n=== upper ear over [{XS[0]:g},{XS[-1]:g}] x '
          f'[{YS[0]:g},{YS[-1]:g}] ===')
    masks = {}
    for buggy in (False, True):
        masks[buggy] = counts[(buggy, True)] > counts[(buggy, False)]
        lab = 'with the bug restored' if buggy else 'current code'
        print(f'  {lab:22s} area {masks[buggy].sum()*cell:.2f} sq units '
              f'({masks[buggy].sum()} of {masks[buggy].size} cells)')
    lost = masks[False].sum() - masks[True].sum()
    print(f'  the bug erased {lost} cells = '
          f'{100*lost/max(masks[False].sum(), 1):.0f}% of the ear')

    print('\n=== which column the bug damages ===')
    X, Y = np.meshgrid(XS, YS)
    for tied in (True, False):
        diff = counts[(False, tied)] != counts[(True, tied)]
        lab = 'FULL' if tied else 'UNIFORM'
        print(f'  {lab:8s} differs in {diff.sum():3d} of {diff.size} cells '
              f'({100*diff.mean():.1f}%)', end='')
        if diff.any():
            print(f'   x {X[diff].min():.2f}-{X[diff].max():.2f}, '
                  f'|y| {np.abs(Y[diff]).min():.2f}-{np.abs(Y[diff]).max():.2f}')
        else:
            print()
    print('\n  (The bug fires in BOTH columns -- it drops the same target -- but '
          'it only\n   changes the COUNT under uniform weight: with rho = (0, 1) '
          'the far-target\n   commitment becomes trivially self-consistent, which '
          'FULL weighting already\n   had and uniform weight correctly did not.)')


if __name__ == '__main__':
    main()
