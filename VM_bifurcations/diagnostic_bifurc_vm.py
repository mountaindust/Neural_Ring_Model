"""
Diagnostic for the unexpected 4/5-stable and 0-stable-island regions reported
in the last cell of compare_sc_vm.ipynb (vonmises model, two circle targets at
(4.33, +-2.5), r=0.5, k=0.55).

Strategy
--------
1. Reproduce the bifurcation diagram on a coarse 41x41 grid (no refinement)
   and save the count map and (x,y) grid as .npy so we can pick representative
   points.
2. At a representative point inside each suspect region, recompute Im/Re of
   dgamma_dt(R+0j, theta) on a fine theta mesh, plot it, and find every root
   independently of gamma_equilib's solver. Verify residuals by polishing.
3. For each candidate (theta_eq, R_eq) verify stability two ways:
     - the analytical _discrim_A criterion that gamma_equilib uses, and
     - by integrating the *coupled* ODE (gamma + heading) starting near the
       fixed point and watching whether it returns or escapes.
4. Print a per-point report.

Run with:
    python diagnostic_bifurc_vm.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, root
from scipy.integrate import solve_ivp

import decision_model as model

# ----- 1. Reproduce the user's setup -----
target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
percep = model.PerceptionModel(targets, (0, 0), 0,
                               neural_weight='vonmises',
                               neural_angle='integral')
percep.k = 0.55
nbm = model.NeuralBandModel(percep)


def count_at(xy):
    angles, stab = nbm.sc_equilib(focal_loc=np.array(xy))
    return int(sum(stab)), len(angles), list(zip(angles, stab))


# ----- 2. Coarse grid scan to localize regions -----
xlim = (0.0, 6.0)
ylim = (-3.5, 3.5)
nx, ny = 41, 41
xs = np.linspace(xlim[0], xlim[1], nx)
ys = np.linspace(ylim[0], ylim[1], ny)

grid = np.full((ny, nx), -1, dtype=int)
total_eqs = np.full((ny, nx), -1, dtype=int)

print("Scanning 41x41 grid (no refinement)...")
for j, y in enumerate(ys):
    for i, x in enumerate(xs):
        n_stab, n_total, _ = count_at((x, y))
        grid[j, i] = n_stab
        total_eqs[j, i] = n_total
    print(f"  row {j+1}/{ny} (y={y:+.3f}): "
          f"counts seen so far = {sorted(set(grid[grid >= 0].tolist()))}")

np.save('diagnostic_grid_stable.npy', grid)
np.save('diagnostic_grid_total.npy', total_eqs)
np.save('diagnostic_xs.npy', xs)
np.save('diagnostic_ys.npy', ys)

print()
print("Stable count value distribution:")
unique, counts = np.unique(grid, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u:2d} stable equilibria: {c:4d} grid cells")

print()
print("Total equilibrium count distribution:")
unique, counts = np.unique(total_eqs, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u:2d} equilibria total : {c:4d} grid cells")


# ----- 3. Pick representative points in each suspect region -----
def representative_points():
    """Return a list of (label, (x, y), expected_n_stable) tuples."""
    pts = []
    for n_target in (0, 4, 5):
        idx_y, idx_x = np.where(grid == n_target)
        if len(idx_x) == 0:
            print(f"  no points found with {n_target} stable equilibria")
            continue
        # Pick the median location for a representative interior point
        ic = len(idx_x) // 2
        x, y = xs[idx_x[ic]], ys[idx_y[ic]]
        pts.append((f"n_stab={n_target}", (x, y), n_target))
        # Also pick min/max y to catch symmetric islands
        y_min_i = np.argmin(ys[idx_y])
        y_max_i = np.argmax(ys[idx_y])
        if y_min_i != ic:
            pts.append((f"n_stab={n_target} (low-y)",
                        (xs[idx_x[y_min_i]], ys[idx_y[y_min_i]]), n_target))
        if y_max_i != ic and y_max_i != y_min_i:
            pts.append((f"n_stab={n_target} (high-y)",
                        (xs[idx_x[y_max_i]], ys[idx_y[y_max_i]]), n_target))
    return pts


# ----- 4. Independent root-finding sweep -----
def independent_root_sweep(focal_loc, R_probe=0.5, n_theta=2001):
    """Sweep theta on [-pi, pi], record Im(dgamma_dt) and Re(dgamma_dt)-R, find
    all sign changes of both, polish each candidate with hybr, verify residual,
    deduplicate, and return list of (theta_eq, R_eq, residual, stable) tuples."""
    theta = np.linspace(-np.pi, np.pi, n_theta)
    im = np.array([nbm.dgamma_dt(gamma=R_probe + 0j, focal_angle=t,
                                 focal_loc=focal_loc).imag for t in theta])

    # Sign changes of Im at R_probe -> candidate thetas
    candidates = []
    for i in range(n_theta - 1):
        if im[i] * im[i + 1] < 0:
            try:
                tc = brentq(lambda t: nbm.dgamma_dt(
                    gamma=R_probe + 0j, focal_angle=t,
                    focal_loc=focal_loc).imag, theta[i], theta[i + 1])
                candidates.append(tc)
            except ValueError:
                pass
    for extra in (0.0, np.pi, -np.pi):
        candidates.append(extra)

    eqs = []
    for tc in candidates:
        sol = root(nbm._self_consistent_eq, [tc, R_probe],
                   args=(focal_loc,), method='hybr', tol=1e-12)
        if not sol.success:
            continue
        teq = model.convert_angles(sol.x[0])
        Req = sol.x[1]
        if Req < 0.01 or Req > 1.0:
            continue
        residual = nbm.dgamma_dt(gamma=Req + 0j, focal_angle=teq,
                                 focal_loc=focal_loc)
        if abs(residual) > 1e-7:
            continue
        # Dedup
        if any(abs(model.convert_angles(teq - e[0])) < 1e-3 for e in eqs):
            continue
        stab = nbm._discrim_A(Req + 0j, teq, focal_loc)
        eqs.append((teq, Req, abs(residual), stab))

    eqs.sort(key=lambda e: e[0])
    return theta, im, eqs


# ----- 5. Coupled ODE stability check -----
def coupled_ode_check(focal_loc, theta_eq, R_eq, dtheta_perturb=0.05,
                      t_final=400, K=1.0):
    """Integrate the coupled (gamma, theta) system starting from
    (gamma = R_eq * exp(i * eps_phase), theta = theta_eq + dtheta_perturb).
    Return final theta and final |gamma - R_eq*exp(0j)| in the heading frame.
    Coupling: dtheta_dt = K * R * sin(ego_angle) where ego_angle is from
    convert_gamma. dgamma_dt uses focal_angle = current theta."""
    def rhs(t, y):
        gr, gi, th = y
        gamma = gr + 1j * gi
        dg = nbm.dgamma_dt(gamma=gamma, focal_angle=th, focal_loc=focal_loc)
        ego, R = nbm.convert_gamma(gamma)
        dth = K * R * np.sin(ego)
        return [dg.real, dg.imag, dth]

    y0 = [R_eq, 0.0, theta_eq + dtheta_perturb]
    sol = solve_ivp(rhs, [0, t_final], y0, method='LSODA', rtol=1e-9,
                    atol=1e-11)
    gr, gi, th = sol.y[:, -1]
    th_final = model.convert_angles(th)
    R_final = abs(gr + 1j * gi)
    ego_final, _ = nbm.convert_gamma(gr + 1j * gi)
    drift = abs(model.convert_angles(th_final - theta_eq))
    return th_final, R_final, ego_final, drift


# ----- 6. Loop over representative points and write a report -----
report_path = 'diagnostic_report.txt'
with open(report_path, 'w') as fh:
    pts = representative_points()
    print(f"\nInvestigating {len(pts)} representative points...\n")
    fh.write(f"Investigating {len(pts)} representative points\n")
    fh.write(f"  geometry: 2 circle targets r=0.5 at (4.33, +-2.5)\n")
    fh.write(f"  perception: vonmises k=0.55, integral neural-angle\n")
    fh.write(f"  NBM defaults: T={nbm.T}, K={nbm.K}\n\n")
    for (label, (x, y), expected) in pts:
        focal_loc = np.array([x, y])
        theta, im, eqs = independent_root_sweep(focal_loc)
        n_stab_indep = sum(1 for e in eqs if e[3])
        n_total_indep = len(eqs)

        block = []
        block.append(f"=== {label} at ({x:+.3f}, {y:+.3f}) ===")
        block.append(f"  bifurc-grid stable count : {expected}")
        block.append(f"  independent total eqs    : {n_total_indep}")
        block.append(f"  independent stable count : {n_stab_indep}")
        block.append("  per-equilibrium results:")
        for (teq, Req, res, stab) in eqs:
            block.append(f"    theta={teq:+.4f} R={Req:.4f} "
                         f"|resid|={res:.2e} stable(_discrim_A)={stab}")
            # Coupled ODE check (only if stab claimed True, plus 1 unstable
            # for sanity)
            if stab:
                th_final, R_final, ego_final, drift = coupled_ode_check(
                    focal_loc, teq, Req, dtheta_perturb=0.08)
                block.append(f"      coupled-ODE: heading drift "
                             f"|dth|={drift:.4f}, "
                             f"R_final={R_final:.3f}, "
                             f"ego_final={ego_final:+.4f}")
        block.append("")
        text = "\n".join(block)
        print(text)
        fh.write(text + "\n")

print(f"\nReport written to {report_path}")
print(f"Grid saved to diagnostic_grid_stable.npy / diagnostic_grid_total.npy")
