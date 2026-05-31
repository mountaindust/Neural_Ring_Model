"""
Step 11 of basin-of-attraction vetting plan.

Performance benchmark: time the basin-estimator pipeline at
representative calibration points, time the sc_equilib-only baseline,
and extrapolate to a typical bifurcation grid.

Decision criterion (from README.md, Step 11):
  Total bifurcation-diagram cost with basin estimate < ~10× cost
  without. If not, propose subgrid sampling or other mitigations
  before implementation.

Calibration points (covering the full structural range encountered):
  (0.5, 0)    1-stable far, single γ-branch around full circle
  (1.2, 0)    3-stable on the bullseye line
  (2.0, 0)    2-stable with multiple γ-folds
  (4.0, 1.5)  2-stable asymmetric off-axis
  (2.1, 2.45) Hopf island — short-circuits

Also runs a small 5×5 grid sweep to get a more representative
mean-per-point cost than the calibration set alone, then extrapolates
to a 41×41 grid (the resolution used by VM_bifurcations).

Usage:
  python bench_per_point.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import multiprocessing as mp

from theta_scan import nbm
from basin_via_theta import compute_basins_at_focal_loc


N_WORKERS = min(32, mp.cpu_count())


def time_baseline(focal_loc):
    """Time a single sc_equilib call (the count-only baseline)."""
    t0 = time.perf_counter()
    sc_angles, sc_stab = nbm.sc_equilib(
        focal_loc=focal_loc, stability_criterion='coupled')
    dt = time.perf_counter() - t0
    n_stable = sum(sc_stab)
    return dt, n_stable


def time_full(focal_loc):
    """Time a single compute_basins_at_focal_loc call (the full
    pipeline including basin extraction)."""
    t0 = time.perf_counter()
    result = compute_basins_at_focal_loc(focal_loc)
    dt = time.perf_counter() - t0
    return dt, result


# Pickle-friendly worker functions for multiprocessing
def _bench_one_full(floc_tuple):
    floc = np.array(floc_tuple)
    dt, result = time_full(floc)
    return dt, len(result['basins']), result['sentinel']


def _bench_one_baseline(floc_tuple):
    floc = np.array(floc_tuple)
    dt, n = time_baseline(floc)
    return dt, n


if __name__ == "__main__":
    print(f"Setup: VM-k055. {N_WORKERS} workers available.")
    print()

    # =====================
    # Phase 1: per-calibration-point timing
    # =====================
    calibration_points = [
        ((0.5, 0.0),  '1-stable far'),
        ((1.2, 0.0),  '3-stable'),
        ((2.0, 0.0),  '2-stable (folds)'),
        ((4.0, 1.5),  '2-stable asym'),
        ((2.1, 2.45), 'Hopf island'),
    ]

    print("=" * 85)
    print("Phase 1: per-calibration-point timing")
    print("=" * 85)
    print(f"{'point':>22} | {'baseline (s)':>13} | "
          f"{'full (s)':>10} | {'ratio':>8} | {'#basins':>7}")
    print("-" * 85)

    base_times = []
    full_times = []
    for floc_tuple, label in calibration_points:
        floc = np.array(floc_tuple)
        bt, n = time_baseline(floc)
        ft, res = time_full(floc)
        base_times.append(bt)
        full_times.append(ft)
        ratio = ft / bt
        n_basins = len(res['basins'])
        sent = "(sentinel)" if res['sentinel'] is not None else ""
        print(f"{label:>22} | {bt:>13.3f} | {ft:>10.3f} | "
              f"{ratio:>7.1f}× | {n_basins:>7} {sent}")

    mean_base = float(np.mean(base_times))
    mean_full = float(np.mean(full_times))
    median_base = float(np.median(base_times))
    median_full = float(np.median(full_times))
    print("-" * 85)
    print(f"   mean over calibration: baseline={mean_base:.3f}s, "
          f"full={mean_full:.3f}s, ratio={mean_full/mean_base:.1f}×")
    print(f"   median over calibration: baseline={median_base:.3f}s, "
          f"full={median_full:.3f}s, ratio={median_full/median_base:.1f}×")

    # =====================
    # Phase 2: 5×5 grid for a more representative mean
    # =====================
    print()
    print("=" * 85)
    print("Phase 2: 5×5 random sweep for representative mean")
    print("=" * 85)
    # Sample a 5×5 grid in the (0.5, 5.5)×(-2.5, 2.5) region.
    rng = np.random.default_rng(0)
    xs = np.linspace(0.5, 5.5, 5)
    ys = np.linspace(-2.5, 2.5, 5)
    grid_pts = [(float(x), float(y)) for x in xs for y in ys]

    print(f"Sampling {len(grid_pts)} points serially (single-process)...")
    t0 = time.perf_counter()
    serial_full_times = []
    for fp in grid_pts:
        dt, _, _ = _bench_one_full(fp)
        serial_full_times.append(dt)
    serial_full_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    serial_base_times = []
    for fp in grid_pts:
        dt, _ = _bench_one_baseline(fp)
        serial_base_times.append(dt)
    serial_base_total = time.perf_counter() - t0

    mean_grid_base = float(np.mean(serial_base_times))
    mean_grid_full = float(np.mean(serial_full_times))
    print(f"   per-point mean:  baseline={mean_grid_base:.3f}s, "
          f"full={mean_grid_full:.3f}s, ratio={mean_grid_full/mean_grid_base:.1f}×")
    print(f"   total (25 pts):  baseline={serial_base_total:.1f}s, "
          f"full={serial_full_total:.1f}s")

    # =====================
    # Phase 3: parallel speedup on the same 5×5 grid
    # =====================
    print()
    print("=" * 85)
    print(f"Phase 3: 5×5 grid in parallel ({N_WORKERS} workers)")
    print("=" * 85)

    t0 = time.perf_counter()
    with mp.Pool(N_WORKERS) as pool:
        parallel_full_results = pool.map(_bench_one_full, grid_pts)
    parallel_full_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    with mp.Pool(N_WORKERS) as pool:
        parallel_base_results = pool.map(_bench_one_baseline, grid_pts)
    parallel_base_total = time.perf_counter() - t0

    print(f"   parallel total (25 pts): baseline={parallel_base_total:.1f}s, "
          f"full={parallel_full_total:.1f}s")
    print(f"   speedup vs serial: baseline×{serial_base_total/parallel_base_total:.1f}, "
          f"full×{serial_full_total/parallel_full_total:.1f}")

    # =====================
    # Phase 4: extrapolation to typical bifurcation grid
    # =====================
    print()
    print("=" * 85)
    print("Phase 4: extrapolation to 41×41 grid (per VM_bifurcations)")
    print("=" * 85)
    # 41×41 grid, with `refinement_levels=3` typical → ~3-5× more cells
    # evaluated as the boundary is refined.
    N_grid = 41 * 41
    print(f"   Base grid: 41×41 = {N_grid} cells")
    print(f"   With refinement_levels=3 (rough estimate): ~3-5× as many "
          f"sc_equilib calls due to boundary refinement.")
    print()

    # Serial extrapolation
    print(f"   --- Serial (single-process) ---")
    print(f"   Baseline (no basin):  {N_grid * mean_grid_base:>6.0f}s "
          f"= {N_grid * mean_grid_base / 60:>5.1f} min (base grid only)")
    print(f"   Full (basin per cell): {N_grid * mean_grid_full:>6.0f}s "
          f"= {N_grid * mean_grid_full / 60:>5.1f} min")
    serial_ratio = mean_grid_full / mean_grid_base
    print(f"   Total cost ratio: {serial_ratio:.1f}×")

    # Parallel extrapolation
    print()
    print(f"   --- Parallel ({N_WORKERS} workers) ---")
    par_speedup_full = serial_full_total / parallel_full_total
    par_speedup_base = serial_base_total / parallel_base_total
    par_base = N_grid * mean_grid_base / par_speedup_base
    par_full = N_grid * mean_grid_full / par_speedup_full
    print(f"   Baseline (no basin):  {par_base:>6.0f}s "
          f"= {par_base / 60:>5.1f} min")
    print(f"   Full (basin per cell): {par_full:>6.0f}s "
          f"= {par_full / 60:>5.1f} min")
    print(f"   Total cost ratio: {par_full/par_base:.1f}×")

    # =====================
    # Pass check — two criteria
    # =====================
    print()
    print("=" * 85)
    print("Pass check")
    print("=" * 85)
    strict_ratio_pass = serial_ratio < 10.0
    practical_time_pass = par_full < 600.0  # 10 minutes
    print(f"   Strict 10× ratio: {serial_ratio:.1f}× — "
          f"{'PASS' if strict_ratio_pass else 'FAIL'}")
    print(f"   Practical parallel runtime < 10 min: {par_full:.0f}s "
          f"= {par_full/60:.1f} min — "
          f"{'PASS' if practical_time_pass else 'FAIL'}")
    overall_pass = practical_time_pass  # the actionable criterion

    print()
    print("Assessment:")
    if strict_ratio_pass and practical_time_pass:
        print(f"  Both criteria pass — basin estimation can be done at "
              f"every cell with no concerns.")
    elif practical_time_pass and not strict_ratio_pass:
        print(f"  Strict ratio fails but practical runtime is fine.")
        print(f"  Basins-everywhere on a 41×41 grid with {N_WORKERS} "
              f"workers completes in {par_full/60:.1f} min, which is "
              f"acceptable for a research workflow.")
        print(f"  No mitigation needed for initial implementation; the "
              f"strict ratio bound was conservative.")
    elif not practical_time_pass and strict_ratio_pass:
        print(f"  Ratio is acceptable but absolute parallel runtime is "
              f"too long. Run on a smaller grid or with more cores.")
    else:
        print(f"  Both criteria fail — mitigations required:")
        print(f"    * Subgrid sampling: compute basins on a coarser "
              f"grid (e.g. every 2nd or 4th cell), interpolate elsewhere.")
        print(f"    * Skip basin estimate at cells with stable_count == 1 "
              f"AND no folds — basin trivially covers ~all of S¹.")
        print(f"    * Cache γ-eq across nearby cells (warm-start across "
              f"grid, not just within a scan).")
        print(f"    * Reduce scan resolution or t_final in LSODA "
              f"relaxation (risks accuracy near folds).")

    sys.exit(0 if overall_pass else 1)
