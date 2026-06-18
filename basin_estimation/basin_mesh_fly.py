"""Fly 3-target variant of the single-panel basin+bifurcation prototype.

Reuses the basin_mesh machinery unchanged but swaps in the fruit-fly GODM
model from walker_analysis/three_target_fly.py: three circle targets 40°
apart at radius 5 (r=0.5), lin_cutoff warp/weight (a_warp=0.65π, b_warp=0.92π,
a_weight=0.20π, b_weight=0.80π), K=2, T=0.10. Purely to see how the current
wheel/placement approach adapts to a busier bifurcation space — nothing
about the visualization is changed.

The basin code references `theta_scan.nbm` (the VM-k055 calibration model)
as a module global; we leave that file untouched and instead inject the fly
model into the four modules that hold the `nbm` / `target_locs` globals, then
call the (now region-parameterized) basin_mesh.main(). The injection happens
before main() forks its worker pool, so the workers inherit the fly model.

Run:  python basin_mesh_fly.py
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'walker_analysis'))

import numpy as np
from three_target_fly import build_model, target_locs as fly_locs

import theta_scan
import basin_via_theta
import basin_arcs
import basin_mesh

fly_nbm = build_model()

# inject the fly model wherever the basin code holds the relevant globals
for mod in (theta_scan, basin_via_theta, basin_arcs, basin_mesh):
    if hasattr(mod, 'nbm'):
        mod.nbm = fly_nbm
    if hasattr(mod, 'target_locs'):
        mod.target_locs = fly_locs

# sanity check: every module that holds `nbm` now points at the fly model,
# and the patched compute path really uses it
assert all(m.nbm is fly_nbm for m in
           (theta_scan, basin_via_theta, basin_arcs, basin_mesh)), \
    "fly model not injected into every module"
_chk = sorted(np.round(np.degrees(basin_mesh.stable_dirs((4.5, 0.0))), 1))
_ref = sorted(np.round(np.degrees(
    [a for a, s in zip(*fly_nbm.sc_equilib(focal_loc=(4.5, 0.0),
                                           stability_criterion='reduced')) if s]), 1))
assert _chk == _ref, f"model injection incomplete: {_chk} != {_ref}"
print(f"fly model active (stable dirs @ (4.5,0): {_chk})")


if __name__ == "__main__":
    # region covers the walker arena: origin out to the targets (x up to 5,
    # y up to ~3.2). Denser base grid than VM-k055 since there are more
    # bifurcation regions to sample.
    # ny odd so y=0 is sampled -> on-axis components get an on-axis rep
    # (keeps the placement mirror-symmetric for this symmetric geometry)
    basin_mesh.main(
        xlim=(0.0, 5.4), ylim=(-3.7, 3.7), bg_res=(73, 85),
        out_name='basin_mesh_fly.png',
        model_label='fly3 — GODM, 40°, radius 5',
        validation_point=None, placement='region')
