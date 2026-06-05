"""
Refined diagnostic: compute the full coupled (gamma_real, gamma_imag, theta)
3x3 Jacobian at each self-consistent equilibrium and compare to _discrim_A,
then map out the disagreement on a moderate grid.

Goal
----
Test the hypothesis that the 4/5 "stable" regions are inflated by saddle
points: equilibria that are gamma-stable at fixed heading (so _discrim_A is
True) but unstable in the coupled (gamma, theta) system because the slow
heading dynamics push the trajectory toward a neighboring true stable
equilibrium.

Also probe the 0-stable islands by integrating the coupled ODE from many
initial headings to see whether some other attractor (limit cycle, drifting
heading) exists.
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

# ---- setup matching compare_sc_vm.ipynb ----
target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
percep = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='vonmises',
                               angle_weight='neural_angle_dist',
                               a_warp=0.55)
nbm = model.NeuralBandModel(percep)
K = nbm.K


def equilibria_at(focal_loc, R_probe=0.5):
    """Use gamma_equilib's logic with a finer theta mesh to find self-consistent
    eqs.  Return list of (theta_eq, R_eq, residual)."""
    theta = np.linspace(-np.pi, np.pi, 2001)
    im = np.array([nbm.dgamma_dt(gamma=R_probe + 0j, focal_angle=t,
                                 focal_loc=focal_loc).imag for t in theta])
    candidates = []
    for i in range(len(theta) - 1):
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
        if any(abs(model.convert_angles(teq - e[0])) < 1e-3 for e in eqs):
            continue
        eqs.append((teq, Req, abs(residual)))
    eqs.sort(key=lambda e: e[0])
    return eqs


def coupled_rhs(y, focal_loc):
    gr, gi, th = y
    gamma = gr + 1j * gi
    dg = nbm.dgamma_dt(gamma=gamma, focal_angle=th, focal_loc=focal_loc)
    ego, R = nbm.convert_gamma(gamma)
    dth = K * R * np.sin(ego)
    return np.array([dg.real, dg.imag, dth])


def jacobian_3d(focal_loc, theta_eq, R_eq, h=1e-6):
    """Numerical 3x3 Jacobian of (gr, gi, th) -> (dgr/dt, dgi/dt, dth/dt) at
    (R_eq, 0, theta_eq)."""
    y0 = np.array([R_eq, 0.0, theta_eq])
    f0 = coupled_rhs(y0, focal_loc)
    J = np.zeros((3, 3))
    for k in range(3):
        yp = y0.copy(); yp[k] += h
        ym = y0.copy(); ym[k] -= h
        J[:, k] = (coupled_rhs(yp, focal_loc) - coupled_rhs(ym, focal_loc)) / (2 * h)
    eigs = np.linalg.eigvals(J)
    return J, eigs


def coupled_stable(focal_loc, theta_eq, R_eq):
    """True if all three eigenvalues of the coupled Jacobian have negative real
    part."""
    _, eigs = jacobian_3d(focal_loc, theta_eq, R_eq)
    return np.all(np.real(eigs) < -1e-8), eigs


def integrate_coupled(focal_loc, init_theta, init_gamma=0.05 + 0j,
                      t_final=400):
    def rhs(t, y):
        return coupled_rhs(y, focal_loc)
    y0 = [init_gamma.real, init_gamma.imag, init_theta]
    sol = solve_ivp(rhs, [0, t_final], y0, method='LSODA', rtol=1e-9,
                    atol=1e-11, dense_output=False)
    return sol


# -------------------------------------------------------------------------
# Part 1: Detailed report at the previously identified suspect points
# -------------------------------------------------------------------------
print("=" * 70)
print("Part 1: coupled Jacobian at suspect points")
print("=" * 70)

points = [
    ("5-stable centre", (1.500, 0.000)),
    ("0-stable upper", (2.100, +2.450)),
    ("0-stable lower", (2.100, -2.450)),
    # additional sanity points
    ("origin (1-stable expected)", (0.0, 0.0)),
    ("near targets (1.0, 0.0)", (1.0, 0.0)),
    ("between (3.0, 0.0)", (3.0, 0.0)),
]

with open("diagnostic_coupled_report.txt", "w") as fh:
    for label, xy in points:
        focal_loc = np.array(xy)
        eqs = equilibria_at(focal_loc)
        block = [f"\n{label} at ({xy[0]:+.3f}, {xy[1]:+.3f})",
                 f"  found {len(eqs)} self-consistent equilibria"]
        n_disc = 0
        n_coupled = 0
        for (teq, Req, res) in eqs:
            disc_stable = nbm._discrim_A(Req + 0j, teq, focal_loc)
            cstable, eigs = coupled_stable(focal_loc, teq, Req)
            n_disc += int(disc_stable)
            n_coupled += int(cstable)
            eig_str = ", ".join(f"{e.real:+.4f}{e.imag:+.4f}j"
                                for e in eigs)
            block.append(f"    theta={teq:+.4f}  R={Req:.4f}  "
                         f"_discrim_A={disc_stable}  "
                         f"coupled-stable={cstable}\n"
                         f"      eigs(J3): [{eig_str}]")
        block.append(f"  TOTAL: _discrim_A={n_disc}  "
                     f"coupled-stable={n_coupled}")
        text = "\n".join(block)
        print(text)
        fh.write(text + "\n")

# -------------------------------------------------------------------------
# Part 2: Probe the 0-stable island for non-equilibrium attractors
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Part 2: integrate coupled ODE from many headings inside the "
      "0-stable island")
print("=" * 70)

focal_loc = np.array([2.100, 2.450])
print(f"\nfocal_loc = {focal_loc}")
init_thetas = np.linspace(-np.pi, np.pi, 25, endpoint=False)
finals = []
for th0 in init_thetas:
    sol = integrate_coupled(focal_loc, th0, init_gamma=0.05 + 0j,
                            t_final=600)
    final_th = model.convert_angles(sol.y[2, -1])
    final_g = sol.y[0, -1] + 1j * sol.y[1, -1]
    final_R = abs(final_g)
    final_ego, _ = nbm.convert_gamma(final_g)
    # residual of dgamma_dt at the final point with focal_angle = final_th
    final_dg = nbm.dgamma_dt(gamma=final_g, focal_angle=final_th,
                             focal_loc=focal_loc)
    finals.append((th0, final_th, final_R, final_ego, abs(final_dg)))
    print(f"  init_theta={th0:+.4f}  ->  final_theta={final_th:+.4f}  "
          f"R={final_R:.3f}  ego={final_ego:+.4f}  "
          f"|dgamma|={abs(final_dg):.2e}")

# Are all final headings the same?
final_thetas = np.array([f[1] for f in finals])
print(f"\n  final theta range: {final_thetas.min():+.4f} to "
      f"{final_thetas.max():+.4f}")
print(f"  spread: {final_thetas.max() - final_thetas.min():.4f}")

# Plot the time series of theta for one trajectory
sol = integrate_coupled(focal_loc, 0.0, init_gamma=0.05 + 0j, t_final=2000)
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
axes[0].plot(sol.t, model.convert_angles(sol.y[2]))
axes[0].set_ylabel("theta")
axes[1].plot(sol.t, np.abs(sol.y[0] + 1j * sol.y[1]))
axes[1].set_ylabel("|gamma|")
ego = []
for k in range(sol.y.shape[1]):
    e, _ = nbm.convert_gamma(sol.y[0, k] + 1j * sol.y[1, k])
    ego.append(e)
axes[2].plot(sol.t, ego)
axes[2].set_ylabel("ego_angle")
axes[2].set_xlabel("t")
fig.suptitle(f"coupled ODE from theta_0=0 at focal_loc={focal_loc}")
fig.tight_layout()
fig.savefig("diagnostic_zero_island_trajectory.png", dpi=120)
print("\n  trajectory saved to diagnostic_zero_island_trajectory.png")
