"""Coupled vs reduced stability comparison.

For two configurations -- (A) the documented vonmises k=0.55 two-target
"island" setup, and (B) a standard smooth-cutoff (a=0, b=pi, uniform weight)
setup -- scan observer (x,y) locations, count stable self-consistent
equilibria under both the full 3x3 'coupled' criterion and the
timescale-separated 'reduced' criterion, and report where they differ.

Expectation:
  - Standard cutoff (no Hopf): reduced == coupled everywhere.
  - vonmises island: they differ inside Hopf region(s), where the coupled 3x3
    sees an oscillatory (Hopf) instability the slaved reduction omits. The
    reduced count is >= coupled count there (an unstable focus + limit cycle
    becomes a stable node under the slaved view).

Produces side-by-side PNGs and a printed disagreement summary.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

from decision_model import Targets, PerceptionModel, NeuralBandModel
from parallel_config import get_n_workers

OUTDIR = os.path.dirname(os.path.abspath(__file__))
TWO_CIRCLE = np.array([[4.33, 2.5], [4.33, -2.5]])


def _make_model(kind):
    t = Targets(locs=TWO_CIRCLE, geom_name='circle', r=0.5)
    if kind == 'vonmises':
        pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='vonmises',
                             angle_weight='neural_angle_dist', a_warp=0.55)
    elif kind == 'cutoff':
        pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='cutoff',
                             a_warp=0.0, b_warp=np.pi, angle_weight=None)
    else:
        raise ValueError(kind)
    return NeuralBandModel(pm)


def _worker(args):
    """Count stable SC equilibria under both criteria at one (x, y)."""
    kind, x, y = args
    nbm = _make_model(kind)
    fl = np.array([x, y])
    _, s_red = nbm.sc_equilib(focal_loc=fl, stability_criterion='reduced')
    _, s_cpl = nbm.sc_equilib(focal_loc=fl, stability_criterion='coupled')
    return x, y, int(sum(s_red)), int(sum(s_cpl))


def scan(kind, xlim, ylim, nx, ny, pool):
    xs = np.linspace(*xlim, nx)
    ys = np.linspace(*ylim, ny)
    args = [(kind, float(x), float(y)) for y in ys for x in xs]
    results = pool.map(_worker, args)
    red = np.zeros((ny, nx), int)
    cpl = np.zeros((ny, nx), int)
    for x, y, r, c in results:
        i = int(np.argmin(np.abs(xs - x)))
        j = int(np.argmin(np.abs(ys - y)))
        red[j, i] = r
        cpl[j, i] = c
    return xs, ys, red, cpl


def plot_panels(kind, xs, ys, red, cpl, fname):
    diff = red - cpl
    vmax = int(max(red.max(), cpl.max(), 1))
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    cmap = plt.get_cmap('viridis', vmax + 1)
    norm = BoundaryNorm(np.arange(-0.5, vmax + 1.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in [(axes[0], red, f'{kind}: reduced'),
                            (axes[1], cpl, f'{kind}: coupled')]:
        im = ax.imshow(data, origin='lower', extent=extent, aspect='equal',
                       interpolation='nearest', cmap=cmap, norm=norm)
        ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, ticks=range(vmax + 1), label='# stable')

    dmax = int(max(abs(diff).max(), 1))
    im2 = axes[2].imshow(diff, origin='lower', extent=extent, aspect='equal',
                         interpolation='nearest', cmap='RdBu_r',
                         vmin=-dmax, vmax=dmax)
    axes[2].set_title(f'{kind}: reduced - coupled')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
    fig.colorbar(im2, ax=axes[2], label='count difference')
    fig.tight_layout()
    out = os.path.join(OUTDIR, fname)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def report(kind, xs, ys, red, cpl):
    diff = red - cpl
    n_disagree = int(np.count_nonzero(diff))
    print(f"\n=== {kind} ===")
    print(f"  grid {red.shape[1]}x{red.shape[0]} = {red.size} cells")
    print(f"  reduced stable-count histogram: "
          f"{dict(zip(*np.unique(red, return_counts=True)))}")
    print(f"  coupled stable-count histogram: "
          f"{dict(zip(*np.unique(cpl, return_counts=True)))}")
    print(f"  disagreeing cells: {n_disagree} "
          f"({100*n_disagree/red.size:.1f}%)")
    if n_disagree:
        js, is_ = np.nonzero(diff)
        print(f"  difference values (reduced-coupled): "
              f"{dict(zip(*np.unique(diff[js, is_], return_counts=True)))}")
        # a few representative disagreeing locations
        print("  sample disagreeing (x, y, reduced, coupled):")
        for k in range(min(8, n_disagree)):
            j, i = js[k], is_[k]
            print(f"    ({xs[i]:.3f}, {ys[j]:.3f})  reduced={red[j,i]}  "
                  f"coupled={cpl[j,i]}")
    return n_disagree


if __name__ == '__main__':
    with Pool(get_n_workers()) as pool:
        # (B) standard cutoff: expect zero disagreement
        xs, ys, red, cpl = scan('cutoff', (0.5, 5.0), (-3.0, 3.0),
                                nx=37, ny=49, pool=pool)
        report('cutoff', xs, ys, red, cpl)
        plot_panels('cutoff', xs, ys, red, cpl, 'compare_cutoff.png')

        # (A) vonmises island: focus on the documented upper-island region.
        xs, ys, red, cpl = scan('vonmises', (1.0, 3.5), (1.2, 3.4),
                                nx=51, ny=45, pool=pool)
        report('vonmises', xs, ys, red, cpl)
        plot_panels('vonmises', xs, ys, red, cpl, 'compare_vonmises_island.png')

        # also a broad vonmises view for context
        xs, ys, red, cpl = scan('vonmises', (0.5, 5.0), (-3.0, 3.0),
                                nx=37, ny=49, pool=pool)
        report('vonmises (broad)', xs, ys, red, cpl)
        plot_panels('vonmises_broad', xs, ys, red, cpl,
                    'compare_vonmises_broad.png')
    print("\nDone. PNGs in", OUTDIR)
