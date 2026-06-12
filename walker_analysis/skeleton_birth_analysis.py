"""Mechanism study: WHEN (in observer x, on the y=0 cut) are the stable consensus
directions born, and WHY are the center and outer-target births separated?

On the midline the three-target cascade (fly default) is:
  x~0.90  compromise arms (+-22 deg) BORN by saddle-node (center still stable)
  x~1.55  center (0 deg) DESTABILIZES (subcritical pitchfork: absorbs 2 saddles)
  x~2.8   center REBORN stable (pitchfork: transverse eigenvalue of theta=0 -> -)
  x~3.05  outer-target branches (+-74 deg) BORN by saddle-node (marginal/flickering)

So "center stable" (a pitchfork re-stabilization of the always-present symmetric
theta=0 equilibrium) happens at a smaller x than "outer stable" (a genuine
saddle-node creating an off-axis, outer-locked consensus). This script sweeps
parameters / kernels / target counts on the y=0 cut to find what sets each birth
location and the gap between them.

Run:  python walker_analysis/skeleton_birth_analysis.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_model as model
import walker_analysis.decision_skeleton as ds

pi = np.pi
cm = model.convert_angles


# --------------------------------------------------------------------------- #
# General model builder (arbitrary kernel, params, targets)
# --------------------------------------------------------------------------- #
def ring(n, sep_deg, radius, phase_deg=0.0):
    """n targets `sep_deg` apart on a circle of `radius`, centred on the +x axis."""
    k = np.arange(n)
    # symmetric about +x axis: angles centred on 0
    ang = np.radians(phase_deg) + np.radians(sep_deg) * (k - (n - 1) / 2.0)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def three_targets(sep_deg=40.0, radius=5.0):
    return ring(3, sep_deg, radius)


def ninegon(radius=5.0, sep_deg=40.0):
    """9 targets `sep_deg` apart at fixed radius (full ring when sep=40)."""
    k = np.arange(9)
    ang = np.radians(sep_deg) * k
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def make_model(locs, *, r=0.5, warp='lin_cutoff', weight='lin_cutoff',
               a_warp=0.47 * pi, b_warp=0.92 * pi,
               a_weight=0.40 * pi, b_weight=0.80 * pi, T=0.2, K=2.0):
    targets = model.Targets(locs=np.asarray(locs, float), geom_name='circle', r=r)
    kw = dict(neural_angle_dist=warp, angle_weight=weight)
    # warp slots
    if warp in ('lin_cutoff', 'cutoff', 'symmetric_beta', 'reg_power'):
        kw.update(a_warp=a_warp, b_warp=b_warp)
    elif warp == 'vonmises':
        kw.update(a_warp=a_warp)
    elif warp == 'direct_power':
        kw.update(a_warp=a_warp)
    # weight slots
    if weight in ('lin_cutoff', 'cutoff', 'symmetric_beta', 'reg_power'):
        kw.update(a_weight=a_weight, b_weight=b_weight)
    elif weight == 'vonmises':
        kw.update(a_weight=a_weight)
    elif weight == 'neural_angle_dist' or weight is None:
        pass
    pm = model.PerceptionModel(targets, (0, 0), 0, **kw)
    return model.NeuralBandModel(pm, T=T, K=K)


# --------------------------------------------------------------------------- #
# Landmark detector on the y=0 cut (jitter-robust)
# --------------------------------------------------------------------------- #
def _stable_thetas(nm, x):
    a, R, s = ds.sc_equilib_with_R(nm, (x, 0.0))
    return [(cm(t), r) for t, r, k in zip(a, R, s) if k]


def landmarks(nm, xmax, dx=0.02, center_tol_deg=6.0, outer_min_deg=50.0,
              persist=4):
    """Find the key birth/death x-locations on the y=0 cut.

    Returns a dict with: arm_birth, center_destab, center_reborn, outer_birth
    (np.nan if not found). 'outer_birth' uses a persistence window (the outer
    saddle-node is marginal and the solver flickers there), reported as the first x
    whose forward `persist`-window has the outer branch present at least twice."""
    xs = np.arange(0.05, xmax + 1e-9, dx)
    has_center = np.zeros(len(xs), bool)
    has_arm = np.zeros(len(xs), bool)
    has_outer = np.zeros(len(xs), bool)
    ctol = np.radians(center_tol_deg)
    omin = np.radians(outer_min_deg)
    for i, x in enumerate(xs):
        th = _stable_thetas(nm, x)
        has_center[i] = any(abs(t) < ctol for t, _ in th)
        has_arm[i] = any(ctol <= abs(t) < omin for t, _ in th)
        has_outer[i] = any(abs(t) >= omin for t, _ in th)

    def first_true(mask, start=0):
        idx = np.where(mask[start:])[0]
        return xs[start + idx[0]] if idx.size else np.nan

    arm_birth = first_true(has_arm)
    # center destabilization: first x where center is absent after arm_birth
    out = dict(arm_birth=arm_birth)
    if not np.isnan(arm_birth):
        i0 = int(np.argmin(np.abs(xs - arm_birth)))
        lost = np.where(~has_center[i0:])[0]
        cd = xs[i0 + lost[0]] if lost.size else np.nan
        out['center_destab'] = cd
        if not np.isnan(cd):
            i1 = int(np.argmin(np.abs(xs - cd)))
            back = np.where(has_center[i1:])[0]
            out['center_reborn'] = xs[i1 + back[0]] if back.size else np.nan
        else:
            out['center_reborn'] = np.nan
    else:
        out['center_destab'] = np.nan
        out['center_reborn'] = np.nan

    # outer birth (persistence-filtered onset)
    ob = np.nan
    for i in range(len(xs) - persist):
        if has_outer[i] and has_outer[i:i + persist].sum() >= 2:
            ob = xs[i]
            break
    out['outer_birth'] = ob
    out['_xs'] = xs
    out['_has'] = dict(center=has_center, arm=has_arm, outer=has_outer)
    return out


def diagnostics_at(nm, x, heading):
    """Neural angles, weights, R of the visible targets at a given heading on y=0,
    plus the SC residual -- to inspect what the configuration looks like at a birth."""
    pm = nm.percep_model
    neur, rho = pm.get_neural_signals(heading, np.array([x, 0.0]))
    # relax gamma at this fixed heading to read R
    g = nm.run_dgamma_dt(focal_angle=heading, focal_loc=np.array([x, 0.0]),
                         init_gamma=0.4 + 0j)
    return dict(neur_deg=np.degrees(neur), rho=rho, R=abs(g), N=neur.size,
                heading_deg=np.degrees(heading))


if __name__ == '__main__':
    # smoke test: reproduce the fly default landmarks
    nm = make_model(three_targets(40, 5), r=0.5)
    lm = landmarks(nm, xmax=4.2)
    print('fly default landmarks (deg/x):')
    for k in ('arm_birth', 'center_destab', 'center_reborn', 'outer_birth'):
        print(f'  {k:14s} = {lm[k]:.2f}')
    print('  gap (outer - center_reborn) = %.2f'
          % (lm['outer_birth'] - lm['center_reborn']))
