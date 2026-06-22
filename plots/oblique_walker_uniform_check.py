"""Sanity check: oblique circle-target walkers with UNIFORM weight.

Identical to the circle-target walker panel in ``oblique_walker.py`` (lin_cutoff
warp a=pi/8,b=pi; r=0.25 circles at the re-oriented near/far locations; K=2,
sigma=0.5 constant noise, v=0.2, dt=0.05, 100 reps, fixed seed) EXCEPT the
angle_weight role is None (uniform) instead of the neural-density weight. The
point is to confirm the walker picture is essentially unchanged by the weight
choice for these extended targets. The walker setup is imported from
oblique_walker (single source of truth) -- only the weight differs.

Run:  python plots/oblique_walker_uniform_check.py
"""
import os

import oblique_walker as ow

OUT = os.path.join(ow.HERE, 'oblique_walker_uniform_check.png')


def main():
    fig, ax = ow.plt.subplots(figsize=(9, 5))
    with ow.Pool(ow.get_n_workers()) as pool:
        ow.render_walkers(ax, pool, geom='circle', radius=ow.WALK_TARGET_R,
                          title='Circle targets, uniform weight', weight=None)
    fig.suptitle('Uniform-weight check: oblique circle-target walkers',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    ow.plt.close(fig)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
