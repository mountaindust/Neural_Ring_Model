"""
Step 10 of basin-of-attraction vetting plan.

Verify graceful failure at a Hopf-island focal_loc, where there are
no stable SC equilibria — the actual attractor is a stable limit cycle.

`basin_estimation.basin_via_theta.compute_basins_at_focal_loc` should
return an empty basin list with a sentinel string noting the situation,
rather than crashing, hanging, or returning garbage.

Calibration: (2.1, 2.45) in VM-k055, the Hopf island documented in
VM_bifurcations/VERDICT.md (sc_equilib reports 0 stable + 1 unstable
focus; a stable limit cycle of period ~17 is the actual attractor).

Tests:
  T1: at (2.1, 2.45) — returns 0 basins with a sentinel that mentions
      "Hopf" or "no stable" or "limit cycle".
  T2: at (1.2, 0) — returns 3 basins with no sentinel (sanity check
      that the wrapper doesn't fail on normal multistable cases).
  T3: at (0.5, 0) — returns 1 basin with no sentinel (sanity check
      on the trivial 1-stable case).

Usage:
  python hopf_island_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from basin_via_theta import compute_basins_at_focal_loc


def _summary(result):
    print(f"    stable_count = {result['stable_count']}, "
          f"unstable_count = {result['unstable_count']}, "
          f"basins = {len(result['basins'])}")
    if result['sentinel'] is not None:
        print(f"    sentinel: {result['sentinel']}")


if __name__ == "__main__":
    results = []

    # ---- T1: Hopf island ----
    print("=" * 65)
    print("T1: Hopf island at (2.1, 2.45) — expect 0 basins + sentinel")
    print("=" * 65)
    focal_loc = np.array([2.1, 2.45])
    t0 = time.time()
    res = compute_basins_at_focal_loc(focal_loc)
    elapsed = time.time() - t0
    _summary(res)
    print(f"    wall time: {elapsed:.2f}s")
    sentinel_ok = (res['sentinel'] is not None
                    and any(kw in res['sentinel'].lower()
                            for kw in ('hopf', 'no stable', 'limit cycle')))
    t1_pass = (res['stable_count'] == 0
                and len(res['basins']) == 0
                and sentinel_ok
                and elapsed < 30)
    print(f"    T1: {'PASS' if t1_pass else 'FAIL'}")
    results.append(('T1 Hopf island returns empty + sentinel', t1_pass))

    # ---- T2: 3-stable point ----
    print()
    print("=" * 65)
    print("T2: 3-stable at (1.2, 0) — expect 3 basins, no sentinel")
    print("=" * 65)
    focal_loc = np.array([1.2, 0.0])
    t0 = time.time()
    res = compute_basins_at_focal_loc(focal_loc)
    elapsed = time.time() - t0
    _summary(res)
    print(f"    wall time: {elapsed:.2f}s")
    t2_pass = (res['stable_count'] == 3
                and len(res['basins']) == 3
                and res['sentinel'] is None
                and elapsed < 30)
    print(f"    T2: {'PASS' if t2_pass else 'FAIL'}")
    results.append(('T2 3-stable returns 3 basins, no sentinel', t2_pass))

    # ---- T3: 1-stable point ----
    print()
    print("=" * 65)
    print("T3: 1-stable at (0.5, 0) — expect 1 basin, no sentinel")
    print("=" * 65)
    focal_loc = np.array([0.5, 0.0])
    t0 = time.time()
    res = compute_basins_at_focal_loc(focal_loc)
    elapsed = time.time() - t0
    _summary(res)
    print(f"    wall time: {elapsed:.2f}s")
    t3_pass = (res['stable_count'] == 1
                and len(res['basins']) == 1
                and res['sentinel'] is None
                and elapsed < 30)
    print(f"    T3: {'PASS' if t3_pass else 'FAIL'}")
    results.append(('T3 1-stable returns 1 basin, no sentinel', t3_pass))

    # ---- Summary ----
    print()
    print("=" * 65)
    n_pass = sum(1 for _, p in results if p)
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n{n_pass}/{len(results)} tests passed.")
    sys.exit(0 if n_pass == len(results) else 1)
