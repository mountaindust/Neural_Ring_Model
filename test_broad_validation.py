"""Broad validation of NeuralBandModel vs IsingExtModel equilibria.

Compares self-consistent equilibria at a grid of spatial points for:
1. Delta targets, no warping
2. Circle targets (r=0.5), no warping

Then examines:
3. Delta targets, power warping c=0.5
4. Circle targets, power warping c=0.5
"""

from decision_model import (Targets, PerceptionModel, NeuralBandModel,
                            IsingExtModel, convert_angles)
from multiprocessing import Pool
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

locs = np.array([[15, 5], [15, 15]])

# Spatial grid matching the standard bifurcation diagram
xs = np.linspace(1, 14, 14)
ys = np.linspace(1, 19, 17)
grid_locs = [(x, y) for x in xs for y in ys]


def _compare_worker(args):
    """Worker for compare_models. Solves both NBM and IEM at one grid point."""
    loc, geom, r, neural_angle = args
    tgts = Targets(locs=locs, geom_name=geom, r=r)
    pm = PerceptionModel(tgts, focal_loc=loc, focal_angle=0,
                         neural_weight='cutoff', neural_angle=neural_angle)
    nbm = NeuralBandModel(pm)
    iem = IsingExtModel(pm)

    nbm_angles, nbm_stab = nbm.gamma_equilib(focal_angle=True, focal_loc=loc)
    nbm_angles = np.array(nbm_angles)
    nbm_stab = np.array(nbm_stab)

    gammas = iem.gamma_equilib(focal_angle=True, focal_loc=loc)
    if not gammas:
        iem_angles = np.array([])
        iem_stab = np.array([], dtype=bool)
    else:
        iem_angles = np.array([np.angle(g) for g in gammas])
        iem_stab = np.array([iem._discrim_A_nu(g, loc) for g in gammas])

    return loc, nbm_angles, nbm_stab, iem_angles, iem_stab


def compare_models(geom, r, label, neural_angle=None, pool=None):
    """Compare NBM and IEM at all grid points. Return mismatch count."""
    mismatches = 0

    t0 = time.time()
    args_list = [(loc, geom, r, neural_angle) for loc in grid_locs]
    if pool is not None:
        results = pool.map(_compare_worker, args_list)
    else:
        results = [_compare_worker(a) for a in args_list]

    for loc, nbm_angles, nbm_stab, iem_angles, iem_stab in results:
        nbm_n_stable = int(nbm_stab.sum()) if len(nbm_stab) else 0
        iem_n_stable = int(iem_stab.sum()) if len(iem_stab) else 0

        nbm_stable_angles = sorted(nbm_angles[nbm_stab]) if nbm_n_stable else []
        iem_stable_angles = sorted(iem_angles[iem_stab]) if iem_n_stable else []

        match = True
        if nbm_n_stable != iem_n_stable:
            match = False
        elif nbm_n_stable > 0:
            if not np.allclose(nbm_stable_angles, iem_stable_angles, atol=0.05):
                match = False

        if not match:
            mismatches += 1
            print(f'  MISMATCH at ({loc[0]:.0f},{loc[1]:.0f}): '
                  f'NBM {nbm_n_stable}S angles={np.round(nbm_stable_angles,3)} '
                  f'IEM {iem_n_stable}S angles={np.round(iem_stable_angles,3)}')

    elapsed = time.time() - t0
    total = len(results)
    print(f'  {label}: {total} points, {mismatches} mismatches, {elapsed:.1f}s')
    return mismatches


def _warping_worker(args):
    """Worker for examine_warping. Solves NBM at one grid point."""
    loc, geom, r, c = args
    tgts = Targets(locs=locs, geom_name=geom, r=r)
    pm = PerceptionModel(tgts, focal_loc=loc, focal_angle=0,
                         neural_weight='cutoff', neural_angle='power')
    pm.c = c
    nbm = NeuralBandModel(pm)

    angles, stab = nbm.gamma_equilib(focal_angle=True, focal_loc=loc)
    angles = np.array(angles)
    stab = np.array(stab)
    n_stable = int(stab.sum()) if len(stab) else 0
    return loc, n_stable


def examine_warping(geom, r, label, c=0.5, pool=None):
    """Examine equilibria with power warping. Report stability counts."""
    n_0stable = 0
    n_1stable = 0
    n_2stable = 0
    n_3plus_stable = 0
    zero_stable_locs = []

    t0 = time.time()
    args_list = [(loc, geom, r, c) for loc in grid_locs]
    if pool is not None:
        results = pool.map(_warping_worker, args_list)
    else:
        results = [_warping_worker(a) for a in args_list]

    for loc, n_stable in results:
        if n_stable == 0:
            n_0stable += 1
            zero_stable_locs.append(loc)
        elif n_stable == 1:
            n_1stable += 1
        elif n_stable == 2:
            n_2stable += 1
        else:
            n_3plus_stable += 1

    elapsed = time.time() - t0
    total = len(results)
    print(f'  {label}: {total} points, {elapsed:.1f}s')
    print(f'    0-stable: {n_0stable}  1-stable: {n_1stable}  '
          f'2-stable: {n_2stable}  3+-stable: {n_3plus_stable}')
    if zero_stable_locs:
        print(f'    0-stable locations: {zero_stable_locs}')
    return n_0stable


if __name__ == '__main__':
    # HW-TEMP: 4-core laptop; restore to 10 on main workstation
    with Pool(4) as pool:
        # ============================================================
        print('=== Validation: NeuralBandModel vs IsingExtModel (no warping) ===')
        # ============================================================

        print('\n--- Delta targets ---')
        m1 = compare_models(None, None, 'delta', pool=pool)

        print('\n--- Circle targets (r=0.5) ---')
        m2 = compare_models('circle', 0.5, 'circle', pool=pool)

        print(f'\nTotal mismatches (no warping): {m1 + m2}')

        # ============================================================
        print('\n\n=== Warping examination: power c=0.5 ===')
        # ============================================================

        print('\n--- Delta targets, power warping ---')
        z1 = examine_warping(None, None, 'delta warping', pool=pool)

        print('\n--- Circle targets (r=0.5), power warping ---')
        z2 = examine_warping('circle', 0.5, 'circle warping', pool=pool)

        print(f'\nTotal 0-stable points: delta={z1}, circle={z2}')
