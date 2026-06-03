"""
Two follow-up investigations.

Q1: Confirm numerically that "saddles" in the coupled 3-eq system are exact
    equilibria of that system (i.e. dtheta/dt = 0 there too, not just ~0).

Q2: Characterise the 0-stable island.
    - Compute the geometric configuration: distances, allocentric angles,
      angular extents, blocking status of each target.
    - Map the 0-stable island boundary at fine resolution.
    - Long-time integration with phase-portrait + power-spectrum to determine
      whether the attractor is a limit cycle, drifting heading, or something
      else.
    - Track how the equilibrium structure changes as we cross the island
      boundary (which bifurcation produces the island?).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, root
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

import decision_model as model

# ---- setup ----
target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
percep = model.PerceptionModel(targets, (0, 0), 0,
                               neural_angle_dist='vonmises',
                               angle_weight='neural_angle_dist',
                               a_warp=0.55)
nbm = model.NeuralBandModel(percep)
K = nbm.K


def find_eqs(focal_loc, R_probe=0.5):
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
        eqs.append((teq, Req))
    return eqs


def coupled_rhs(y, focal_loc):
    gr, gi, th = y
    gamma = gr + 1j * gi
    dg = nbm.dgamma_dt(gamma=gamma, focal_angle=th, focal_loc=focal_loc)
    ego, R = nbm.convert_gamma(gamma)
    dth = K * R * np.sin(ego)
    return np.array([dg.real, dg.imag, dth])


def coupled_eigs(focal_loc, theta_eq, R_eq, h=1e-6):
    y0 = np.array([R_eq, 0.0, theta_eq])
    J = np.zeros((3, 3))
    for k in range(3):
        yp = y0.copy(); yp[k] += h
        ym = y0.copy(); ym[k] -= h
        J[:, k] = (coupled_rhs(yp, focal_loc) -
                   coupled_rhs(ym, focal_loc)) / (2 * h)
    return J, np.linalg.eigvals(J)


# =========================================================================
# Q1: Saddles really are 3-equation equilibria
# =========================================================================
print("=" * 72)
print("Q1: Verify that saddle points are exact equilibria of the 3-eq system")
print("=" * 72)
points_to_check = [
    ("5-stable centre", (1.500, 0.000)),
    ("3-stable nearby", (1.000, 0.000)),
    ("misclassified strip", (3.800, 1.500)),
]
for label, xy in points_to_check:
    focal_loc = np.array(xy)
    eqs = find_eqs(focal_loc)
    print(f"\n{label} at {xy}: {len(eqs)} self-consistent equilibria")
    for (teq, Req) in eqs:
        rhs = coupled_rhs(np.array([Req, 0.0, teq]), focal_loc)
        # rhs[0,1] are dgamma_re, dgamma_im at gamma=R+0j (the self-consistent
        # search guarantees these vanish). rhs[2] is dtheta/dt — should also
        # be machine zero because ego_angle = inv_neural(0) = 0.
        ego_at_eq, _ = nbm.convert_gamma(Req + 0j)
        print(f"  theta={teq:+.4f} R={Req:.4f}  ->  "
              f"|dgamma|={np.hypot(rhs[0], rhs[1]):.2e}, "
              f"|dtheta|={abs(rhs[2]):.2e}, ego_angle={ego_at_eq:+.2e}")


# =========================================================================
# Q2a: Geometry of the 0-stable island representative point
# =========================================================================
print("\n" + "=" * 72)
print("Q2a: Geometry at the 0-stable island point (2.10, +2.45)")
print("=" * 72)
focal_loc = np.array([2.10, 2.45])
for i, (xt, yt) in enumerate(target_locs):
    dx = xt - focal_loc[0]
    dy = yt - focal_loc[1]
    dist = np.hypot(dx, dy)
    bearing = np.arctan2(dy, dx)            # allocentric, since heading=0
    half_extent = np.arcsin(targets.r / dist)
    print(f"  target {i} at ({xt}, {yt}):  "
          f"dist={dist:.3f},  bearing={np.degrees(bearing):+.2f} deg,  "
          f"half-ext={np.degrees(half_extent):.2f} deg")

# Use _get_target_signals at theta=-0.0876 (the unstable equilibrium heading)
theta_eq = -0.0876
neur_angles, rho = percep.get_neural_signals(focal_angle=theta_eq,
                                              focal_loc=focal_loc)
print(f"\n  At equilibrium heading theta={theta_eq:+.4f}:")
print(f"    visible targets: {len(neur_angles)}  "
      f"(0 = both blocked by each other; "
      f"this is what _get_target_signals returns after blocking)")
for i, (na, r) in enumerate(zip(neur_angles, rho)):
    print(f"    target {i}: neur_angle={na:+.4f}  rho={r:.4f}")
print(f"    sum(rho)={rho.sum():.4f}")


# =========================================================================
# Q2b: Long-time integration to identify attractor type
# =========================================================================
print("\n" + "=" * 72)
print("Q2b: Long-time dynamics inside the 0-stable island")
print("=" * 72)

def integrate(focal_loc, init_theta, init_gamma=0.05 + 0j, t_final=4000):
    def rhs(t, y):
        return coupled_rhs(y, focal_loc)
    y0 = [init_gamma.real, init_gamma.imag, init_theta]
    return solve_ivp(rhs, [0, t_final], y0, method='LSODA',
                     rtol=1e-10, atol=1e-12, max_step=2.0)


sol = integrate(focal_loc, init_theta=0.0, t_final=4000)
gr, gi, th = sol.y
gamma_arr = gr + 1j * gi
R_arr = np.abs(gamma_arr)
ego_arr = np.array([nbm.convert_gamma(g)[0] for g in gamma_arr])

# Look at the last ~500 time units to see asymptotic behavior
mask = sol.t > sol.t[-1] - 500
t_late = sol.t[mask]
th_late = model.convert_angles(th[mask])
R_late = R_arr[mask]
ego_late = ego_arr[mask]

print(f"  late-time theta range: [{th_late.min():+.5f}, "
      f"{th_late.max():+.5f}]  spread={th_late.max() - th_late.min():.2e}")
print(f"  late-time R range:     [{R_late.min():.5f}, "
      f"{R_late.max():.5f}]  spread={R_late.max() - R_late.min():.2e}")
print(f"  late-time ego range:   [{ego_late.min():+.5f}, "
      f"{ego_late.max():+.5f}]  spread={ego_late.max() - ego_late.min():.2e}")

# Estimate period if it's a limit cycle
peaks, _ = find_peaks(th_late, prominence=1e-8)
if len(peaks) > 1:
    periods = np.diff(t_late[peaks])
    print(f"  detected {len(peaks)} theta-peaks; "
          f"mean period = {periods.mean():.3f} (std {periods.std():.3f})")
else:
    print(f"  no oscillation detected in theta; system may be at fixed point")

# Phase-portrait plot
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes[0, 0].plot(sol.t, model.convert_angles(th))
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('theta')
axes[0, 0].set_title('heading vs time')
axes[0, 1].plot(sol.t, R_arr)
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('|gamma|')
axes[0, 1].set_title('|gamma| vs time')
axes[1, 0].plot(sol.t, ego_arr)
axes[1, 0].set_xlabel('t')
axes[1, 0].set_ylabel('ego_angle')
axes[1, 0].set_title('ego_angle vs time')
# Phase portrait in (theta, ego) — should reveal limit cycle
late_mask = sol.t > 1000
axes[1, 1].plot(model.convert_angles(th[late_mask]), ego_arr[late_mask],
                lw=0.5)
axes[1, 1].plot([theta_eq], [0], 'rx', markersize=12,
                label=f'unstable eq (theta={theta_eq:.3f})')
axes[1, 1].set_xlabel('theta')
axes[1, 1].set_ylabel('ego_angle')
axes[1, 1].set_title('phase portrait, t>1000')
axes[1, 1].legend()
fig.suptitle(f'Long-time dynamics inside 0-stable island '
             f'at focal_loc={focal_loc}')
fig.tight_layout()
fig.savefig('diagnostic_island_long_dynamics.png', dpi=120)
print("  saved diagnostic_island_long_dynamics.png")


# =========================================================================
# Q2c: How the 0-stable island arises — track equilibria & eigenvalues
#      across the boundary.
# =========================================================================
print("\n" + "=" * 72)
print("Q2c: Bifurcation crossing the 0-stable island boundary")
print("=" * 72)

# Sweep along x at y=2.45 — should cross into and out of the island.
xs = np.linspace(0.5, 4.5, 81)
records = []   # list of (x, n_eqs, list of (theta, R, max_re_eig))
for x in xs:
    fl = np.array([x, 2.45])
    eqs = find_eqs(fl)
    eq_data = []
    for (teq, Req) in eqs:
        _, eigs = coupled_eigs(fl, teq, Req)
        eq_data.append((teq, Req, max(np.real(eigs))))
    records.append((x, len(eqs), eq_data))

# Plot eigenvalue signs vs x for each equilibrium
fig2, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax_th = axes[0]
ax_re = axes[1]
for (x, n, eq_data) in records:
    for (teq, Req, max_re) in eq_data:
        color = 'g' if max_re < -1e-8 else 'r'
        ax_th.plot(x, teq, 'o', color=color, markersize=4)
        ax_re.plot(x, max_re, 'o', color=color, markersize=4)
ax_th.axhline(0, color='k', lw=0.4)
ax_th.set_ylabel('theta_eq (green=stable, red=unstable)')
ax_th.set_title(f'Equilibrium structure along y=2.45 slice')
ax_re.axhline(0, color='k', lw=0.4)
ax_re.set_xlabel('x')
ax_re.set_ylabel('max Re(eigenvalue)')
fig2.tight_layout()
fig2.savefig('diagnostic_island_x_slice.png', dpi=120)
print("  saved diagnostic_island_x_slice.png")

# Print summary table where # equilibria changes
print("\n  rows where the equilibrium structure changes:")
prev_n = None
for (x, n, eq_data) in records:
    summary = " | ".join(f"th={t:+.3f},Re={r:+.4f}" for (t, _, r) in eq_data)
    if n != prev_n:
        print(f"    x={x:.3f}  n_eqs={n}  {summary}")
    prev_n = n
