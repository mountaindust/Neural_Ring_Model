"""
Prototype (standalone — does NOT touch decision_model.py): basin-of-
attraction arc widths per stable SC equilibrium, via the fixed
neutral-seed slaved flow. This is the panel-B kernel from TODO.md #5.

Protocol (TODO.md #5, findings.md §14.8a):
  - pin the observer at focal_loc, sweep heading θ around S¹;
  - for each θ, seed γ at a fixed NEUTRAL state: arg(γ)=0 (consensus
    straight ahead / heading-aligned) and R = R_SEED, a small "indecision"
    magnitude (~0.1-0.2: below the committed R≳0.4 seen in walker sims,
    above the R≈0 arg-degeneracy);
  - run the SLAVED flow (re-equilibrate γ warm-started each θ-step, then
    dθ = K·R·sin(arg(γ)/2)) to steady state;
  - record which stable SC eq it lands on.
The basin of a stable eq is the arc of θ that flows to it; its arc width
is the robustness scalar. Boundaries are located by destination-flip
bisection, seeded neutrally on both sides (single-valued — no §14
history-dependence).

Bonus probe ("which destination requires commitment?", per the user):
repeat the map with a COMMITTED seed (R=1, same arg=0) and compare. In a
fold wedge the only knob with arg fixed is R, so where the two maps
disagree is exactly the commitment-sensitive wedge; the destination that
only the committed seed reaches is the one that "requires commitment."
We characterize it by target distance and SC-eq R.

Validates at VM-k055 focal_loc=(4.0, 1.5) (the §9/§14 asymmetric point).

Usage:  python basin_arcs.py [x y]
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from theta_scan import nbm, _relax_gamma_cached, _circular_angle_diff, target_locs
from basin_via_theta import find_sc_gamma

R_SEED = 0.15      # indecision seed magnitude
R_COMMITTED = 1.0  # committed seed magnitude (for the bonus probe)


def wrap(a):
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def _fl(focal_loc):
    """Plain-float tuple for clean titles (avoids np.float64 repr)."""
    return tuple(round(float(v), 4) for v in focal_loc)


# -----------------------------------------------------------------------------
# Slaved deterministic θ-flow (matches NeuralBandModel.plot_walkers, no noise)
# -----------------------------------------------------------------------------
def slaved_flow(focal_loc, gamma0, theta0, dt=0.1, n_steps=2000,
                conv_tol=1e-5, t_final=100):
    """Run the slaved flow from (theta0, gamma0) to steady state.

    γ is re-equilibrated (warm-started from the previous step) at every
    heading step, then dθ = K·R·sin(arg(γ)/2). Returns (theta_f, gamma_f,
    converged).
    """
    gamma = gamma0
    theta = theta0
    converged = False
    for _ in range(n_steps):
        gamma = _relax_gamma_cached(theta, focal_loc, gamma, t_final=t_final)
        R = abs(gamma)
        dtheta = nbm.K * R * np.sin(np.angle(gamma) / 2.0)
        theta = wrap(theta + dtheta * dt)
        if abs(dtheta) < conv_tol:
            converged = True
            break
    return theta, gamma, converged


def stable_sc_eqs(focal_loc, criterion='reduced'):
    """Sorted allocentric headings of the stable SC equilibria."""
    ang, stab = nbm.sc_equilib(focal_loc=focal_loc,
                               stability_criterion=criterion)
    return sorted([a for a, s in zip(ang, stab) if s])


def _label_from_final(theta_f, R_f, stable, converged,
                      max_dist=0.15, min_R=0.05):
    """Map a flow endpoint to a stable-eq index, or -1 (no basin: did not
    converge, collapsed to R≈0, or landed away from any stable eq)."""
    if (not converged) or R_f < min_R or not stable:
        return -1
    dists = [abs(_circular_angle_diff(theta_f, s)) for s in stable]
    j = int(np.argmin(dists))
    return j if dists[j] < max_dist else -1


def destination_label(focal_loc, theta0, stable, R_seed):
    theta_f, gamma_f, conv = slaved_flow(focal_loc, R_seed + 0j, theta0)
    return _label_from_final(theta_f, abs(gamma_f), stable, conv)


# -----------------------------------------------------------------------------
# Basin arcs: coarse destination sweep + destination-flip bisection
# -----------------------------------------------------------------------------
def basin_arcs(focal_loc, R_seed=R_SEED, n_coarse=120, n_bisect=22,
               verbose=True):
    """Return (stable, arcs, widths) where arcs is a list of
    (theta_start, theta_end, label) covering S¹ and widths[label] is the
    total arc measure flowing to that stable eq. label -1 = no basin."""
    stable = stable_sc_eqs(focal_loc)
    if verbose:
        print(f"  stable SC eqs ({len(stable)}): "
              f"{[f'{np.degrees(s):+.1f}°' for s in stable]}")

    # 1-stable cell: the whole circle is one basin (Poincaré-Hopf); no sweep.
    if len(stable) == 1:
        arcs = [(-np.pi, np.pi, 0)]
        return stable, arcs, {0: 2 * np.pi}
    if len(stable) == 0:
        return stable, [(-np.pi, np.pi, -1)], {-1: 2 * np.pi}

    # coarse destination sweep
    thetas = np.linspace(-np.pi, np.pi, n_coarse, endpoint=False)
    labels = np.empty(n_coarse, dtype=int)
    t0 = time.time()
    for i, th in enumerate(thetas):
        labels[i] = destination_label(focal_loc, th, stable, R_seed)
    if verbose:
        print(f"  coarse sweep: {n_coarse} flows in {time.time()-t0:.1f}s")

    # locate every label change (cyclic) and bisect to the separatrix
    step = 2 * np.pi / n_coarse
    seps = []
    for i in range(n_coarse):
        j = (i + 1) % n_coarse
        la, lb = labels[i], labels[j]
        if la == lb:
            continue
        a, b = thetas[i], thetas[i] + step
        for _ in range(n_bisect):
            m = 0.5 * (a + b)
            lm = destination_label(focal_loc, wrap(m), stable, R_seed)
            if lm == la:
                a = m
            else:
                b = m
        seps.append(wrap(0.5 * (a + b)))

    # build arcs between consecutive separatrices; label each by its midpoint
    arcs = []
    widths = {}
    if not seps:
        # all one label
        lab = int(labels[0])
        arcs = [(-np.pi, np.pi, lab)]
        widths[lab] = 2 * np.pi
        return stable, arcs, widths

    seps_sorted = sorted(seps)
    nseps = len(seps_sorted)
    for k in range(nseps):
        s_start = seps_sorted[k]
        s_end = seps_sorted[(k + 1) % nseps]
        span = (s_end - s_start) % (2 * np.pi)
        mid = wrap(s_start + 0.5 * span)
        lab = destination_label(focal_loc, mid, stable, R_seed)
        arcs.append((s_start, wrap(s_start + span), lab))
        widths[lab] = widths.get(lab, 0.0) + span
    return stable, arcs, widths


# -----------------------------------------------------------------------------
# Helpers for reporting / the commitment probe
# -----------------------------------------------------------------------------
def target_geometry(focal_loc):
    """Distance and allocentric direction from focal_loc to each target."""
    d = target_locs - np.asarray(focal_loc, dtype=float)
    dist = np.linalg.norm(d, axis=1)
    alloc = np.arctan2(d[:, 1], d[:, 0])
    return dist, alloc


def describe_stable(focal_loc, stable):
    """For each stable eq: matched target (closest allocentric direction),
    its distance, and the SC-eq R = |γ_sc|."""
    dist, alloc = target_geometry(focal_loc)
    info = []
    for s in stable:
        k = int(np.argmin([abs(_circular_angle_diff(s, a)) for a in alloc]))
        g = find_sc_gamma(s, focal_loc)
        R = abs(g) if g is not None else float('nan')
        info.append({'theta': s, 'target': k, 'dist': dist[k],
                     'alloc': alloc[k], 'R_sc': R})
    return info


def compute_point(focal_loc, R_seed=R_SEED, n_coarse=120, n_bisect=22):
    """Per-point kernel for the panel-B mesh: returns a dict with the
    stable SC directions, their basin arcs, basin widths (the robustness
    scalar), and per-stable descriptive info. Importable; no plotting."""
    stable, arcs, widths = basin_arcs(focal_loc, R_seed, n_coarse=n_coarse,
                                      n_bisect=n_bisect, verbose=False)
    info = describe_stable(focal_loc, stable)
    return {'focal_loc': np.asarray(focal_loc, float), 'stable': stable,
            'arcs': arcs, 'widths': widths, 'info': info}


def commitment_probe(focal_loc, stable, n_coarse=120):
    """Compare uncommitted (R_SEED) vs committed (R_COMMITTED) destination
    maps. Returns (thetas, lab_unc, lab_com, sensitive_mask)."""
    thetas = np.linspace(-np.pi, np.pi, n_coarse, endpoint=False)
    lab_unc = np.array([destination_label(focal_loc, th, stable, R_SEED)
                        for th in thetas])
    lab_com = np.array([destination_label(focal_loc, th, stable, R_COMMITTED)
                        for th in thetas])
    sensitive = (lab_unc != lab_com) & (lab_unc >= 0) & (lab_com >= 0)
    return thetas, lab_unc, lab_com, sensitive


def two_branch_probe(focal_loc, theta_w):
    """At a single heading, relax γ from a low-R and a high-R seed to expose
    the two coexisting γ-branches (the fold structure)."""
    g_lo = _relax_gamma_cached(theta_w, focal_loc, R_SEED + 0j, t_final=200)
    g_hi = _relax_gamma_cached(theta_w, focal_loc, R_COMMITTED + 0j,
                               t_final=200)
    return g_lo, g_hi


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_basin_wheel(focal_loc, stable, arcs, widths, info, out_path):
    """Panel-B prototype at one (x,y): a basin wheel (rim colored by
    destination) with a radial arrow per stable direction, length ∝ basin
    arc width."""
    colors = plt.cm.tab10(np.arange(10))
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    # rim: colored basin arcs
    for (s_start, s_end, lab) in arcs:
        span = (s_end - s_start) % (2 * np.pi)
        c = 'lightgray' if lab < 0 else colors[lab % 10]
        ax.bar(s_start + 0.5 * span, height=0.12, width=span, bottom=1.0,
               color=c, edgecolor='white', linewidth=0.5, align='center')

    # arrows: one per stable direction, length ∝ arc width
    for lab, s in enumerate(stable):
        L = widths.get(lab, 0.0) / (2 * np.pi)  # fraction of the circle
        c = colors[lab % 10]
        ax.annotate('', xy=(s, L), xytext=(s, 0),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=2.5))
        d = info[lab]
        ax.text(s, min(L + 0.08, 1.0),
                f"tgt{d['target']} d={d['dist']:.2f}\n"
                f"R={d['R_sc']:.2f}, {np.degrees(widths.get(lab,0)/1):.0f}",
                ha='center', va='center', fontsize=7.5, color=c)

    ax.set_ylim(0, 1.18)
    ax.set_yticklabels([])
    ax.set_title(f"Basin wheel @ focal_loc={_fl(focal_loc)}  (neutral seed "
                 f"R={R_SEED})\nrim = destination basin; arrow length ∝ basin "
                 f"arc width (robustness)", fontsize=10, pad=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_commitment(focal_loc, thetas, lab_unc, lab_com, sensitive, out_path):
    colors = plt.cm.tab10(np.arange(10))
    deg = np.degrees(thetas)

    def colvec(labs):
        return [('lightgray' if l < 0 else colors[l % 10]) for l in labs]

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.bar(deg, 1.0, width=360/len(thetas), bottom=1.15,
           color=colvec(lab_unc), align='edge')
    ax.bar(deg, 1.0, width=360/len(thetas), bottom=0.0,
           color=colvec(lab_com), align='edge')
    for i, th in enumerate(deg):
        if sensitive[i]:
            ax.axvspan(th, th + 360/len(thetas), ymin=0, ymax=1,
                       color='red', alpha=0.10)
    ax.text(-185, 1.65, 'uncommitted\n(R=%.2f)' % R_SEED, ha='right',
            va='center', fontsize=9)
    ax.text(-185, 0.5, 'committed\n(R=%.1f)' % R_COMMITTED, ha='right',
            va='center', fontsize=9)
    ax.set_xlim(-200, 185)
    ax.set_ylim(-0.1, 2.25)
    ax.set_yticks([])
    ax.set_xlabel('heading θ [deg]')
    ax.set_title(f"Destination vs seed commitment @ {_fl(focal_loc)}  "
                 f"(red = commitment-sensitive wedge)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
# Validation / demo
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        focal_loc = np.array([float(sys.argv[1]), float(sys.argv[2])])
    else:
        focal_loc = np.array([4.0, 1.5])

    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print(f"basin_arcs prototype  |  focal_loc = {_fl(focal_loc)}  (VM-k055)")
    print("=" * 70)

    # --- enumerate stable SC eqs and describe them -----------------------
    n_red = len(stable_sc_eqs(focal_loc, 'reduced'))
    n_cpl = len(stable_sc_eqs(focal_loc, 'coupled'))
    print(f"stable count: reduced={n_red}, coupled={n_cpl}")

    t_all = time.time()
    stable, arcs, widths = basin_arcs(focal_loc, R_SEED)
    info = describe_stable(focal_loc, stable)

    print("\nstable equilibria (neutral-seed basin widths):")
    print(f"  {'θ':>9} {'target':>7} {'dist':>6} {'R_sc':>6} "
          f"{'basin width':>14}")
    for lab, s in enumerate(stable):
        d = info[lab]
        w = np.degrees(widths.get(lab, 0.0))
        print(f"  {np.degrees(s):+8.1f}° {d['target']:>7} {d['dist']:>6.2f} "
              f"{d['R_sc']:>6.3f} {w:>12.1f}°")
    nob = np.degrees(widths.get(-1, 0.0))
    if nob > 0.5:
        print(f"  (no-basin / collapse arc total: {nob:.1f}°)")

    if len(stable) == 2:
        w0 = widths.get(0, 0.0)
        w1 = widths.get(1, 0.0)
        big, small = (max(w0, w1), min(w0, w1))
        print(f"\nbasin ratio (wider/narrower): {big/small:.2f}×  "
              f"(§9 scan figure was 5.35×; §14 dynamical ≈2×)")

    # --- commitment probe -------------------------------------------------
    if len(stable) >= 2:
        print("\n--- commitment probe (uncommitted R=%.2f vs committed R=%.1f)"
              % (R_SEED, R_COMMITTED))
        thetas, lab_unc, lab_com, sensitive = commitment_probe(
            focal_loc, stable)
        nsens = int(sensitive.sum())
        frac = 100.0 * nsens / len(thetas)
        print(f"  commitment-sensitive headings: {nsens}/{len(thetas)} "
              f"({frac:.0f}% of the circle)")
        if nsens:
            # representative sensitive heading
            idx = np.where(sensitive)[0]
            th_w = thetas[idx[len(idx) // 2]]
            u = int(lab_unc[idx[len(idx) // 2]])
            c = int(lab_com[idx[len(idx) // 2]])
            du, dc = info[u], info[c]
            print(f"  in the sensitive wedge (e.g. θ={np.degrees(th_w):+.1f}°):")
            print(f"    uncommitted → eq θ={np.degrees(stable[u]):+.1f}° "
                  f"(target{du['target']}, dist {du['dist']:.2f}, "
                  f"R_sc {du['R_sc']:.3f})")
            print(f"    committed   → eq θ={np.degrees(stable[c]):+.1f}° "
                  f"(target{dc['target']}, dist {dc['dist']:.2f}, "
                  f"R_sc {dc['R_sc']:.3f})")
            verdict = ("FARTHER" if dc['dist'] > du['dist'] else "CLOSER")
            print(f"    ⇒ commitment is required to reach the {verdict} target "
                  f"(target{dc['target']}).")

            g_lo, g_hi = two_branch_probe(focal_loc, th_w)
            print(f"  two coexisting γ-branches at θ={np.degrees(th_w):+.1f}°:")
            print(f"    low-R seed  → γ R={abs(g_lo):.3f}, "
                  f"arg={np.degrees(np.angle(g_lo)):+.1f}°")
            print(f"    high-R seed → γ R={abs(g_hi):.3f}, "
                  f"arg={np.degrees(np.angle(g_hi)):+.1f}°")
        plot_commitment(focal_loc, thetas, lab_unc, lab_com, sensitive,
                        os.path.join(here, 'basin_arcs_commitment.png'))

    # --- plots ------------------------------------------------------------
    plot_basin_wheel(focal_loc, stable, arcs, widths, info,
                     os.path.join(here, 'basin_arcs_wheel.png'))
    print(f"\ntotal time: {time.time()-t_all:.1f}s")
    print(f"plots saved: basin_arcs_wheel.png, basin_arcs_commitment.png")
