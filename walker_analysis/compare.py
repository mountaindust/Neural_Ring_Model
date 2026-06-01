"""Before/after comparison harness for the sin(Theta*/2) + blind-spot change.

Runs the *real* plot_walkers code path (so it exercises the blind-spot search
after the change) and records lost-walker counts, plus a small sc_equilib
stable-count raster for the bifurcation-invariance check.

Usage:
    python walker_analysis/compare.py before   # run on HEAD before the change
    python walker_analysis/compare.py after     # run after applying the change
    python walker_analysis/compare.py diff       # compare the two saved runs

Stats are saved to walker_analysis/stats_<label>.npz. This harness produced the
8/30 -> 2/30 four-delta result and the bit-identical bifurcation raster that
validated the K-doubling invariance; kept here as documentation of the change.
"""
import sys
import os
import json
import warnings

import matplotlib
matplotlib.use('Agg')  # no display
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_model as dm

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0


def two_circle_model(K=2):
    targets = dm.Targets(locs=np.array([(4.33, 2.5), (4.33, -2.5)]),
                         geom_name='circle', r=0.5)
    pm = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                            focal_angle=0.0,
                            neural_angle_dist='cutoff', a_warp=0.0, b_warp=np.pi,
                            angle_weight='neural_angle_dist')
    return dm.NeuralBandModel(percep_model=pm, T=0.2, K=K)


def four_delta_model(K=10):
    targets = dm.Targets(locs=np.array([(4.33, 2.25), (4.33, -2.25),
                               (4.33, 0.75), (4.33, -0.75)]), geom_name=None)
    pm = dm.PerceptionModel(targets=targets, focal_loc=(0.0, 0.0),
                            focal_angle=0.0,
                            neural_angle_dist='cutoff', a_warp=0.0, b_warp=np.pi,
                            angle_weight='neural_angle_dist')
    return dm.NeuralBandModel(percep_model=pm, T=0.2, K=K)


def run_walkers(nbm, *, reps=30, std=0.5, start_loc=(0.0, 0.0),
                start_angle=0.0, max_steps=1500, title='', fname=None):
    """Run plot_walkers, capture lost-walker count from warnings."""
    nbm.rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(5, 5))
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        nbm.plot_walkers(dt=0.1, v=1, std=std, repetitions=reps,
                         max_steps=max_steps, start_loc=start_loc,
                         start_angle=start_angle, plot_tracks=True, ax=ax,
                         title=title)
        lost = sum('reached max_steps' in str(w.message) for w in wlist)
    if fname:
        fig.savefig(os.path.join(HERE, fname), dpi=90, bbox_inches='tight')
    plt.close(fig)
    return {'reps': reps, 'lost': lost}


def stable_count_raster(nbm, nx=9, ny=9):
    xs = np.linspace(0.2, 6.0, nx)
    ys = np.linspace(-3.4, 3.4, ny)
    raster = np.zeros((ny, nx), dtype=int)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            angles, stable = nbm.sc_equilib(focal_loc=(x, y),
                                            stability_criterion='coupled')
            raster[j, i] = int(np.sum(stable)) if len(stable) else 0
    return raster, xs, ys


def main(label):
    print(f"=== run label: {label} (K default = {dm.NeuralBandModel().K}) ===")
    out = {}

    # Dead-zone / blind reproduction: four delta, full cutoff weighting, K=10
    m4 = four_delta_model(K=10)
    s4 = run_walkers(m4, reps=30, std=0.5, max_steps=1500,
                     title=f'4-delta cutoff b=pi ({label})',
                     fname=f'four_delta_{label}.png')
    print(f"four-delta (b=pi): lost {s4['lost']}/{s4['reps']}")
    out['four_delta_lost'] = s4['lost']
    out['four_delta_reps'] = s4['reps']

    # Two circle targets
    m2 = two_circle_model(K=2)
    s2 = run_walkers(m2, reps=30, std=0.5, max_steps=1500,
                     title=f'2-circle ({label})',
                     fname=f'two_circle_{label}.png')
    print(f"two-circle: lost {s2['lost']}/{s2['reps']}")
    out['two_circle_lost'] = s2['lost']
    out['two_circle_reps'] = s2['reps']

    # Bifurcation stable-count raster (two-circle, full weighting)
    rast, xs, ys = stable_count_raster(two_circle_model(K=2))
    print("two-circle stable-count raster:\n", rast)

    np.savez(os.path.join(HERE, f'stats_{label}.npz'),
             raster=rast, xs=xs, ys=ys,
             meta=json.dumps(out))
    print(f"saved stats_{label}.npz")


def diff():
    a = np.load(os.path.join(HERE, 'stats_before.npz'), allow_pickle=True)
    b = np.load(os.path.join(HERE, 'stats_after.npz'), allow_pickle=True)
    ma = json.loads(str(a['meta']))
    mb = json.loads(str(b['meta']))
    print("=== walker stats (before -> after) ===")
    for k in ma:
        if k.endswith('_lost'):
            reps_k = k.replace('_lost', '_reps')
            print(f"  {k:20s}: {ma[k]}/{ma[reps_k]} -> {mb[k]}/{mb[reps_k]}")
    print("=== bifurcation raster equality (must be exact) ===")
    same = np.array_equal(a['raster'], b['raster'])
    print(f"  identical: {same}")
    if not same:
        print("  before:\n", a['raster'])
        print("  after:\n", b['raster'])
        print("  diff:\n", b['raster'] - a['raster'])


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'before'
    if arg == 'diff':
        diff()
    else:
        main(arg)
