"""Stability comparison figure (publication draft).

Two rows x three observer positions in the ``lin_cutoff`` (a=0, b=pi)
two-target "fly2" setup (first row of ``sweep_a_warp_uniform_weight``),
going from 0 to 2 to 3 stable heading-consistent equilibria:

  (a)/(d) 0-stable -- (2.45, 2.95), on the thin 0-stable arc.  The single
          heading-consistent equilibrium is unstable; the slaved walker runs a
          clockwise gamma-bistability relaxation loop (no rest heading).
  (b)/(e) 2-stable -- (2.45, 0.0), on the symmetry axis behind the targets.
  (c)/(f) 3-stable -- (0.70, 0.0), on the symmetry axis, tri-stable region.

Top row    -- neural consensus angle Theta = arg(gamma) vs heading, full
              circle.  The heading-consistent equilibria are the Theta = 0
              crossings.  Panel (a) carries a zoom box marking the heading
              window detailed in panel (d) below.
Bottom row -- the deterministic turning rate dtheta/dt = K * R * sin(Theta/2)
              vs heading (R = |gamma|).  The equilibria are the dtheta/dt = 0
              crossings; stability is the slope there (negative = stable).
              Panel (d) is zoomed to the 0-stable relaxation loop; (e)/(f)
              are full circle.

Axis labels carry the publication notation: varphi for the heading variable in
configuration space, Phi for the observer heading itself, and a starred
Theta^* = Arg gamma^* for the equilibrium consensus angle. The code below keeps
decision_model's names (theta, Theta = angle(gamma)).

Both rows are drawn from the multivalued gamma-branch structure (every
gamma-equilibrium at each heading, colored by gamma-stability: stable lobe vs
unstable saddle).  Each top-row panel carries a bifurcation-diagram inset
marking its observer location; one shared legend serves the whole figure.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import decision_model as model

# Render mathtext in Computer Modern, matching the default math font of a
# LaTeX document. This needs no LaTeX installation (unlike text.usetex).
plt.rcParams['mathtext.fontset'] = 'cm'

# ----------------------------- config -----------------------------
TARGET_LOCS = np.array([[4.33, 2.5], [4.33, -2.5]])   # fly2 geometry
TARGET_R = 0.5
A_WARP, B_WARP = 0.0, np.pi

T_FINAL = 120                   # gamma relaxation time per heading
ZOOM_HALF_DEG = 6.0             # half-width of the 0-stable zoom about theta_sc
N_FULL = 361                    # heading samples for the full-circle panels
R_MIN = 0.05                    # drop the trivial R~0 behind-state
INSET_RECT = [0.035, 0.06, 0.33, 0.33]   # bottom-left, all top-row panels

# (x, y, count label); column 0 is the 0-stable point shown full (top) + zoom (bottom)
P0 = (2.45, 2.95, '0-stable')
PCOLS = [(2.45, 0.00, '2-stable'), (0.70, 0.00, '3-stable')]

# font sizes (bumped for publication legibility)
LABEL_FS = 14
TITLE_FS = 16
TICK_FS = 12
LEGEND_FS = 11
PANEL_FS = 24         # interior panel-letter labels (A-F)
SUPTITLE_FS = 18

# Bifurcation raster for the top-row insets. This is the cache written by
# plots/neural_weight_sweep.py under its DEFAULT config (WARP_FAMILY
# 'lin_cutoff', WEIGHT None), so if it is missing, regenerate it with
#     python plots/neural_weight_sweep.py
# The npz is gitignored, so a fresh clone must run that once.
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '_cache_neural_weight_sweep_lin_cutoff_warp_uniform_weight.npz')
OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'stability_comparison')
DPI = 300
# Publication output: a single high-quality png at 300 dpi, matching the other
# figures in this directory. (Previously a jpg + tif raster pair plus a vector
# pdf; consolidated to one lossless raster.)
OUTPUT_FORMATS = ('png',)


# ----------------------------- helpers -----------------------------

def build(focal_loc):
    targets = model.Targets(locs=TARGET_LOCS, geom_name='circle', r=TARGET_R)
    percep = model.PerceptionModel(
        targets, focal_loc=focal_loc, focal_angle=0,
        neural_angle_dist='lin_cutoff', angle_weight=None,
        a_warp=A_WARP, b_warp=B_WARP)
    return targets, percep, model.NeuralBandModel(percep)


def wrap(a):
    return ((np.asarray(a) + np.pi) % (2 * np.pi)) - np.pi


def branches(nbm, focal_loc, theta_lo, theta_hi, n):
    """All gamma-equilibria (with A-stability) on a fine heading grid -- the
    multivalued branch structure. Returns (theta, Theta, R, A_stable)."""
    ths = np.linspace(theta_lo, theta_hi, n)
    th_out, Theta_out, R_out, stab_out = [], [], [], []
    for th in ths:
        gs, ss = nbm.gamma_equilib(focal_angle=th,
                                   focal_loc=np.asarray(focal_loc),
                                   stability_criterion='discrim_a')
        for g, s in zip(gs, ss):
            th_out.append(th)
            Theta_out.append(np.angle(g))
            R_out.append(abs(g))
            stab_out.append(bool(s))
    return (np.array(th_out), np.array(Theta_out), np.array(R_out),
            np.array(stab_out, dtype=bool))


def limit_cycle(nbm, focal_loc, theta0, t_end=120, dt=0.02, keep_frac=0.5):
    """Integrate the slaved heading flow; return the late-time limit cycle as
    time-ordered (heading, neural Theta, R)."""
    nbm.gamma = 0.3 + 0.0j
    th = theta0
    th_h, Theta_h, R_h = [], [], []
    n = int(t_end / dt)
    for _ in range(n):
        nbm.gamma = nbm.run_dgamma_dt(focal_angle=th,
                                      focal_loc=np.asarray(focal_loc),
                                      init_gamma=nbm.gamma, t_Final=100,
                                      warn=False)
        g = nbm.gamma
        th_h.append(th)
        Theta_h.append(np.angle(g))
        R_h.append(abs(g))
        th = wrap(th + nbm.K * abs(g) * np.sin(np.angle(g) / 2) * dt)
    k0 = int((1 - keep_frac) * n)
    return (wrap(np.array(th_h[k0:])), np.array(Theta_h[k0:]),
            np.array(R_h[k0:]))


def cycle_jump_indices(lc_Theta, thresh_deg=5.0):
    """Representative up- and down-jump indices in a time-ordered cycle."""
    dTh = np.diff(lc_Theta)
    big = np.where(np.abs(dTh) > np.radians(thresh_deg))[0]
    ups = [i for i in big if dTh[i] > 0]
    downs = [i for i in big if dTh[i] < 0]
    iu = ups[len(ups) // 2] if ups else None
    idn = downs[len(downs) // 2] if downs else None
    return iu, idn


def yvals(Theta, R, K, ymode):
    """Map (Theta, R) to the plotted y-quantity."""
    if ymode == 'Theta':
        return np.degrees(Theta)
    return K * R * np.sin(Theta / 2)            # deterministic turning rate


def load_bifurcation():
    if not os.path.exists(CACHE):
        raise SystemExit(
            f"missing bifurcation cache for the insets:\n  {CACHE}\n"
            "regenerate it with:  python plots/neural_weight_sweep.py")
    d = np.load(CACHE, allow_pickle=False)
    fp = json.loads(str(d['fingerprint_json']))
    return d['imgs'][0], fp


# ----------------------------- key & panels -----------------------------

def panel_letter(ax, letter, loc):
    """Interior panel-letter label. loc='ul' (upper-left, lowered to clear the
    descending curve) or 'ur' (upper-right corner)."""
    if loc == 'ul':
        x, y, ha = 0.035, 0.86, 'left'
    else:
        x, y, ha = 0.965, 0.96, 'right'
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_FS,
            fontweight='bold', va='top', ha=ha, zorder=8)


def _bifurcation_inset(ax, targets, focal_loc, img0, fp, rect):
    """Bifurcation-diagram inset marking a single observer location."""
    axins = ax.inset_axes(rect)
    cmap = plt.get_cmap('viridis', 4)
    norm = BoundaryNorm(np.arange(-0.5, 4.5), 4)
    axins.imshow(np.clip(img0, 0, 3), origin='lower',
                 extent=[fp['xlim'][0], fp['xlim'][1],
                         fp['ylim'][0], fp['ylim'][1]],
                 aspect='equal', interpolation='nearest', cmap=cmap, norm=norm)
    targets.plot_targets_to_axis(axins)
    axins.plot(*focal_loc, marker='*', ms=12, color='red', mec='k', mew=0.6)
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title('# stable eq.', fontsize=7)


def draw_panel(ax, nbm, targets, focal_loc, kind, ymode, K, img0, fp,
               theta_sc=None, title='', inset_rect=None):
    """One panel: y = neural Theta (ymode='Theta') or turning rate
    (ymode='flow') vs heading, from the multivalued gamma-branches."""
    if kind == 'relax':
        z_lo = np.radians(np.degrees(theta_sc) - ZOOM_HALF_DEG)
        z_hi = np.radians(np.degrees(theta_sc) + ZOOM_HALF_DEG)
        th_b, Theta_b, R_b, stab_b = branches(nbm, focal_loc, z_lo, z_hi, n=121)
        yb = yvals(Theta_b, R_b, K, ymode)
        lc_th, lc_Th, lc_R = limit_cycle(nbm, focal_loc, np.radians(120.0))
        yc = yvals(lc_Th, lc_R, K, ymode)
        xlim = [np.degrees(z_lo), np.degrees(z_hi)]
        span = yb.max() - yb.min()
        ylim = [yb.min() - 0.10 * span, yb.max() + 0.10 * span]
    else:
        th_b, Theta_b, R_b, stab_b = branches(nbm, focal_loc, -np.pi, np.pi,
                                              n=N_FULL)
        keep = R_b > R_MIN
        th_b, Theta_b, R_b, stab_b = (th_b[keep], Theta_b[keep], R_b[keep],
                                      stab_b[keep])
        yb = yvals(Theta_b, R_b, K, ymode)
        xlim = [-180, 180]
        ylim = [-180, 180] if ymode == 'Theta' else None

    ax.axhline(0, color='0.5', ls='--', lw=1.0, zorder=1)
    ax.scatter(np.degrees(th_b[stab_b]), yb[stab_b], s=9, color='C0', zorder=2)
    ax.scatter(np.degrees(th_b[~stab_b]), yb[~stab_b], s=11, facecolors='none',
               edgecolors='C1', linewidths=0.7, zorder=2)

    if kind == 'relax':
        ax.plot(np.degrees(lc_th), yc, '-', color='0.25', lw=1.3, alpha=0.9,
                zorder=4)
        iu, idn = cycle_jump_indices(lc_Th)
        xc = np.degrees(lc_th)
        aprops = dict(arrowstyle='-|>', color='k', lw=2.0, mutation_scale=20)
        if iu is not None:                      # left fold: walker jumps up
            ax.annotate('', xy=(xc[iu] - 0.32, yc[iu + 1]),
                        xytext=(xc[iu] - 0.32, yc[iu]), arrowprops=aprops,
                        zorder=5)
        if idn is not None:                     # right fold: walker jumps down
            ax.annotate('', xy=(xc[idn] + 0.32, yc[idn + 1]),
                        xytext=(xc[idn] + 0.32, yc[idn]), arrowprops=aprops,
                        zorder=5)

    ang_sc, stab_sc = nbm.sc_equilib(focal_loc=np.asarray(focal_loc))
    for a, s in zip(ang_sc, stab_sc):
        if xlim[0] <= np.degrees(a) <= xlim[1]:
            ax.plot(np.degrees(a), 0.0, 'o', ms=10, mew=1.6,
                    mfc=('C2' if s else 'white'), mec='k', zorder=6)

    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:                                       # symmetric range for the flow
        ymax = np.abs(yb).max()
        ax.set_ylim(-1.12 * ymax, 1.12 * ymax)
    if kind == 'full':
        ax.set_xticks(range(-180, 181, 90))
        if ymode == 'Theta':
            ax.set_yticks(range(-180, 181, 90))
    if title:
        ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    if inset_rect is not None:
        _bifurcation_inset(ax, targets, focal_loc, img0, fp, inset_rect)
    return ang_sc, stab_sc


def main():
    img0, fp = load_bifurcation()

    # No sharex: column 0's top panel is full-circle while its bottom panel is
    # a zoom, so the two cannot share a heading axis.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # ---- column 0: 0-stable.  top = full-circle Theta (+ zoom box);
    #      bottom = zoomed turning rate (the relaxation loop) ----
    x0, y0, lbl0 = P0
    targets, _, nbm = build((x0, y0))
    K = nbm.K
    ang, stab = nbm.sc_equilib(focal_loc=np.asarray((x0, y0)))
    theta_sc = ang[0]
    draw_panel(axes[0][0], nbm, targets, (x0, y0), 'full', 'Theta', K, img0, fp,
               title=lbl0, inset_rect=INSET_RECT)
    draw_panel(axes[1][0], nbm, targets, (x0, y0), 'relax', 'flow', K, img0, fp,
               theta_sc=theta_sc)
    panel_letter(axes[0][0], 'A', 'ul')
    panel_letter(axes[1][0], 'D', 'ur')
    print(f"  (A/D) {lbl0}: {len(ang)} eq, {int(np.sum(stab))} stable")

    # zoom box on (a): heading window of (d), Theta extent of the branches there
    z_lo = np.degrees(theta_sc) - ZOOM_HALF_DEG
    z_hi = np.degrees(theta_sc) + ZOOM_HALF_DEG
    _, Th_z, _, _ = branches(nbm, (x0, y0), np.radians(z_lo), np.radians(z_hi),
                             n=61)
    by0, by1 = np.degrees(Th_z).min() - 4.0, np.degrees(Th_z).max() + 4.0
    axes[0][0].add_patch(Rectangle((z_lo, by0), z_hi - z_lo, by1 - by0,
                                   fill=False, ec='k', lw=1.6, zorder=7))

    # ---- columns 1, 2: 2-stable, 3-stable.  both rows full-circle ----
    for col, (x, y, lbl) in enumerate(PCOLS, start=1):
        lt, lb = 'BC'[col - 1], 'EF'[col - 1]
        targets, _, nbm = build((x, y))
        K = nbm.K
        draw_panel(axes[0][col], nbm, targets, (x, y), 'full', 'Theta', K,
                   img0, fp, title=lbl, inset_rect=INSET_RECT)
        ang, stab = draw_panel(axes[1][col], nbm, targets, (x, y), 'full',
                               'flow', K, img0, fp)
        panel_letter(axes[0][col], lt, 'ul')
        panel_letter(axes[1][col], lb, 'ur')
        print(f"  ({lt}/{lb}) {lbl}: {len(ang)} eq, {int(np.sum(stab))} stable")

    # row y-labels, bottom x-labels
    axes[0][0].set_ylabel(r'neural consensus angle  $\Theta^*=\mathrm{Arg}\,\gamma^*$  [deg]',
                          fontsize=LABEL_FS)
    axes[1][0].set_ylabel(r'turning rate  $d\Phi/dt = K\,R\,\sin(\Theta^*/2)$',
                          fontsize=LABEL_FS)
    for ax in axes[1]:
        ax.set_xlabel(r'heading the observer is facing, $\varphi$  [deg]',
                      fontsize=LABEL_FS)

    # shared legend (branches + heading equilibria) in the upper-right of (A)
    handles = [
        Line2D([], [], marker='o', ls='', color='C0', ms=8,
               label=r'stable $\gamma$-branch'),
        Line2D([], [], marker='o', ls='', mfc='none', mec='C1', ms=8,
               label=r'unstable $\gamma$-branch'),
        Line2D([], [], marker='o', ls='', mfc='C2', mec='k', ms=10,
               label=r'$\Phi$ stable equilibrium'),
        Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=10,
               label=r'$\Phi$ unstable equilibrium'),
    ]
    axes[0][0].legend(handles=handles, loc='upper right', fontsize=LEGEND_FS,
                      framealpha=0.92)

    # the relaxation limit cycle gets its own legend, in panel D (lower-left)
    axes[1][0].legend(
        handles=[Line2D([], [], color='0.25', lw=1.8,
                        label='relaxation limit cycle')],
        loc='lower left', fontsize=LEGEND_FS, framealpha=0.92)

    fig.suptitle(
        r'$\gamma$-equilibria by heading for linear cutoff density '
        r'without weighting', fontsize=SUPTITLE_FS)
    for fmt in OUTPUT_FORMATS:
        out = f'{OUT_BASE}.{fmt}'
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f'wrote {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
