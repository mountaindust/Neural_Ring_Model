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
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

locs = np.array([[15, 5], [15, 15]])

# Spatial grid matching the standard bifurcation diagram
xs = np.linspace(1, 14, 14)
ys = np.linspace(1, 19, 17)

def get_iem_equilibria(iem, focal_loc):
    """Extract angles and stability from IsingExtModel gamma_equilib."""
    gammas = iem.gamma_equilib(focal_angle=True, focal_loc=focal_loc)
    if not gammas:
        return np.array([]), np.array([], dtype=bool)
    angles = np.array([np.angle(g) for g in gammas])
    stab = np.array([iem._discrim_A_nu(g, focal_loc) for g in gammas])
    return angles, stab


def compare_models(geom, r, label, neural_angle=None):
    """Compare NBM and IEM at all grid points. Return mismatch count."""
    tgts = Targets(locs=locs, geom_name=geom, r=r)
    mismatches = 0
    total = 0

    t0 = time.time()
    for x in xs:
        for y in ys:
            loc = (x, y)
            pm = PerceptionModel(tgts, focal_loc=loc, focal_angle=0,
                                neural_weight='cutoff',
                                neural_angle=neural_angle)
            nbm = NeuralBandModel(pm)
            iem = IsingExtModel(pm)

            nbm_angles, nbm_stab = nbm.gamma_equilib(focal_angle=True,
                                                       focal_loc=loc)
            nbm_angles = np.array(nbm_angles)
            nbm_stab = np.array(nbm_stab)

            iem_angles, iem_stab = get_iem_equilibria(iem, loc)

            # Compare: sort both, match by angle proximity
            nbm_idx = np.argsort(nbm_angles) if len(nbm_angles) else []
            iem_idx = np.argsort(iem_angles) if len(iem_angles) else []

            # Count stable equilibria
            nbm_n_stable = int(nbm_stab.sum()) if len(nbm_stab) else 0
            iem_n_stable = int(iem_stab.sum()) if len(iem_stab) else 0

            # Match: same number of stable equilibria, angles close
            nbm_stable_angles = sorted(nbm_angles[nbm_stab]) if nbm_n_stable else []
            iem_stable_angles = sorted(iem_angles[iem_stab]) if iem_n_stable else []

            match = True
            if nbm_n_stable != iem_n_stable:
                match = False
            elif nbm_n_stable > 0:
                if not np.allclose(nbm_stable_angles, iem_stable_angles,
                                   atol=0.05):
                    match = False

            total += 1
            if not match:
                mismatches += 1
                print(f'  MISMATCH at ({x:.0f},{y:.0f}): '
                      f'NBM {nbm_n_stable}S angles={np.round(nbm_stable_angles,3)} '
                      f'IEM {iem_n_stable}S angles={np.round(iem_stable_angles,3)}')

    elapsed = time.time() - t0
    print(f'  {label}: {total} points, {mismatches} mismatches, {elapsed:.1f}s')
    return mismatches


def examine_warping(geom, r, label, c=0.5):
    """Examine equilibria with power warping. Report stability counts."""
    tgts = Targets(locs=locs, geom_name=geom, r=r)
    n_0stable = 0
    n_1stable = 0
    n_2stable = 0
    n_3plus_stable = 0
    zero_stable_locs = []
    total = 0

    t0 = time.time()
    for x in xs:
        for y in ys:
            loc = (x, y)
            pm = PerceptionModel(tgts, focal_loc=loc, focal_angle=0,
                                neural_weight='cutoff',
                                neural_angle='power')
            pm.c = c
            nbm = NeuralBandModel(pm)

            angles, stab = nbm.gamma_equilib(focal_angle=True,
                                              focal_loc=loc)
            angles = np.array(angles)
            stab = np.array(stab)
            n_stable = int(stab.sum()) if len(stab) else 0

            total += 1
            if n_stable == 0:
                n_0stable += 1
                zero_stable_locs.append((x, y))
            elif n_stable == 1:
                n_1stable += 1
            elif n_stable == 2:
                n_2stable += 1
            else:
                n_3plus_stable += 1

    elapsed = time.time() - t0
    print(f'  {label}: {total} points, {elapsed:.1f}s')
    print(f'    0-stable: {n_0stable}  1-stable: {n_1stable}  '
          f'2-stable: {n_2stable}  3+-stable: {n_3plus_stable}')
    if zero_stable_locs:
        print(f'    0-stable locations: {zero_stable_locs}')
    return n_0stable


# ============================================================
print('=== Validation: NeuralBandModel vs IsingExtModel (no warping) ===')
# ============================================================

print('\n--- Delta targets ---')
m1 = compare_models(None, None, 'delta')

print('\n--- Circle targets (r=0.5) ---')
m2 = compare_models('circle', 0.5, 'circle')

print(f'\nTotal mismatches (no warping): {m1 + m2}')

# ============================================================
print('\n\n=== Warping examination: power c=0.5 ===')
# ============================================================

print('\n--- Delta targets, power warping ---')
z1 = examine_warping(None, None, 'delta warping')

print('\n--- Circle targets (r=0.5), power warping ---')
z2 = examine_warping('circle', 0.5, 'circle warping')

print(f'\nTotal 0-stable points: delta={z1}, circle={z2}')
