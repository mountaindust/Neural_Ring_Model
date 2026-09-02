# Neural Ring Model

**Christopher Strickland** — <cstric12@utk.edu>

A mathematical model of spatial decision-making on a neural ring. A single observer (a
locust, a fly, a foraging animal) sits in a plane with several attractive targets around
it and has to pick *one* direction to travel. The competition between targets is resolved
by Ising-type dynamics on a ring of direction-tuned neurons: each visible target recruits
a population of neurons at the direction where it is perceived, those populations compete
through a cosine coupling kernel, and the winner — or the compromise — sets the direction
the observer turns toward.

The interesting behavior is in the *bifurcations*. As the observer moves through the
plane, stable directions appear, merge, and vanish. That is how the model captures
"commit to one target" versus "split the difference," and the abrupt switch between them.

## Getting started

Requires Python 3 with `numpy`, `scipy`, `matplotlib`, and `jupyter` (plus `pytest` to
run the test suite). Clone the repo, start a notebook from the top-level directory, and
`import decision_model as model` — there is nothing to install or build.

```sh
pytest tests/        # confirm everything works
```

Many plots evaluate the model on a grid of observer locations and are parallelized with
`multiprocessing`. Copy [machine_config.template.py](machine_config.template.py) to
`machine_config.py` and set `N_WORKERS` to something sensible for your machine; see
[PARALLEL_CONFIG.md](PARALLEL_CONFIG.md).

## The three main pieces

The model is a pipeline. Each stage takes the previous one as a constructor argument:

```
Targets  ──►  PerceptionModel  ──►  NeuralBandModel
(what is      (what the observer    (what the neurons
 out there)    actually perceives)   decide to do about it)
```

### `Targets` — the world

Holds the target locations and geometry, and answers the purely geometric questions: what
angular arc does each target subtend from a given observer position, which targets block
which, and would a step of the walker run into one.

```python
import numpy as np
import decision_model as model

target_locs = np.array([[4.33, 2.5], [4.33, -2.5]])       # (x, y) of each target
targets = model.Targets(locs=target_locs, geom_name='circle', r=0.5)
```

Three geometries: `'circle'` (radius `r`), `'capsule'` (a line-segment spine of length `l`
with semicircular endcaps of width `w` and orientation `theta`), and `None` for point
("delta") targets with no angular extent. Give targets different attractiveness with
`values=`.

### `PerceptionModel` — the observer's view

Places an observer at `focal_loc` facing `focal_angle` and turns the scene into two
arrays: a **neural angle** and a **neural group size ρ** for each visible target. This is
where occlusion and the biology of the visual field enter. It has two *independent* roles:

- **`neural_angle_dist` (the warp)** — the map from egocentric angle to neural angle.
  Models denser neural representation near the center of the visual field, so directions
  in front are spread out relative to directions behind. Choose `'lin_cutoff'` (the
  default), `'cutoff'`, `'vonmises'`, `'symmetric_beta'`, `'reg_power'`, `'direct_power'`,
  or `None` for no warp.
- **`angle_weight` (the weight)** — the density integrated over each target's visible arc
  to set ρ, i.e. how strongly a target in a given direction is attended to. Same family
  names, plus `'neural_angle_dist'` to reuse the warp, or `None` (the default) for uniform
  weighting.

Family parameters go in the generic slots `a_warp`/`b_warp` and `a_weight`/`b_weight`;
what each slot *means* depends on the family (for `'lin_cutoff'`, `a` is where the plateau
ends and `b` is where the weight reaches zero).

```python
percep_model = model.PerceptionModel(targets, focal_loc=(0, 0), focal_angle=0,
                                     neural_angle_dist='lin_cutoff',
                                     angle_weight='neural_angle_dist',
                                     a_warp=np.pi/4, b_warp=np.pi)

percep_model.plot_neural_weight()               # the warp/weight shape
percep_model.plot_blocked_signals(wb_plot=True) # what the observer sees from here
angles, rho = percep_model.get_neural_signals() # the arrays handed to the ring
```

`focal_loc`, `focal_angle`, and `targets` can be reassigned freely afterward. To change a
warp or weight parameter, **assign the property** (`percep_model.a_warp = 0.55`) so that
the lookup splines rebuild.

### `NeuralBandModel` — the dynamics

Takes the neural angles and group sizes and runs the ring. Its state is a complex order
parameter **γ**: the argument `Θ = arg(γ)` is the consensus direction in neural
coordinates, and the modulus `R = |γ|` is how strongly committed the observer is (R near 0
is undecided, R near 1 is locked on).

```python
neur_model = model.NeuralBandModel(percep_model, beta=10, K=2)
```

- **`beta`** is the inverse neural temperature — cold (large β) is sharp commitment, hot
  (small β) is diffuse. It is a property of the neural ring, *not* of the scene, so it
  does not scale with the number of targets.
- **`K`** is the turning coupling strength: the observer turns at
  `dθ/dt = K·R·sin(Θ/2)`, so it turns hardest when it is both committed and facing the
  wrong way.
- **`angle_distortion_nu`** is optional, and only for reproducing the distorted-cosine
  coupling kernel of Sridhar et al. (2021). Leave it as `None`.

The methods you will actually call:

| method | what it gives you |
|---|---|
| `sc_equilib(focal_loc=...)` | the self-consistent equilibrium headings at one location, each with a stable/unstable flag — heading = consensus, so the observer has stopped turning |
| `plot_direction_mesh(...)` | a quiver plot of those equilibrium directions over a grid of observer positions |
| `plot_bifurcation_diagram(...)` | a 2D map colored by how many *stable* directions exist at each position — the model's main figure, with adaptive refinement at the boundaries |
| `plot_walkers(...)` | simulates an ensemble of stochastic walkers (heading noise plus the turning law) and plots their trajectories |
| `plot_dtheta_dt(...)` | turning rate vs. heading at a fixed location, for reading equilibria off by eye |
| `run_dgamma_dt(...)`, `gamma_equilib(...)` | the underlying γ relaxation and its equilibria at a *fixed* heading |

Stability is reported under `stability_criterion='reduced'` (the default), which assumes
the neural ring is fast relative to the observer's turning — the same assumption the
simulated walker makes.

The grid-based plots accept a `pool=` argument for multiprocessing, and every plotting
method takes `ax=` to draw onto an existing axis and `wb_plot=True` to size the figure for
a notebook.

```python
from multiprocessing import Pool
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
with Pool(10) as pool:
    neur_model.plot_bifurcation_diagram(xlim=(-3, 6), ylim=(-5, 5),
                                        num_x=19, num_y=19,
                                        pool=pool, ax=ax, wb_plot=True)
neur_model.plot_walkers(dt=0.1, v=0.3, std=0.4, repetitions=50, ax=ax, wb_plot=True)
```

## Worked examples

The notebooks in this directory are the place to start — read them in this order:

- [neural_band.ipynb](neural_band.ipynb) — the basic setup, and the effect of warping vs.
  weighting on the equilibrium directions.
- [neural_band_walker.ipynb](neural_band_walker.ipynb) — walker ensembles drawn on top of
  a bifurcation diagram.
- [compare_sc_vm.ipynb](compare_sc_vm.ipynb) and
  [compare_sc_beta.ipynb](compare_sc_beta.ipynb) — comparing warp families (linear cutoff
  vs. von Mises, and vs. symmetric beta).

## Also in this repo

- [theory/](theory/) — derivations: basins of attraction, the free-energy functional, and
  background notes on Lyapunov/Langevin/Kramers theory.
- [weighting_analysis/](weighting_analysis/) — a study of what angular weighting can and
  cannot do to the observer's choices.
- [tests/](tests/) — the test suite; see [tests/README.md](tests/README.md).
- [plots/](plots/), [walker_analysis/](walker_analysis/) — scripts that generate figures.
- [CLAUDE.md](CLAUDE.md) — an in-depth technical reference on the model internals,
  coordinate conventions, and known limitations.
- [archive/](archive/), [Matlab/](Matlab/) — retired code, kept for reference only.

## License

Copyright © 2026 Christopher Strickland.

This program is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).
