# walker_analysis

Walker (physical-space, x–y) random-walk analysis for `NeuralBandModel`.

Here the noise is angular heading noise and the object of
study is the walker's trajectory in physical space via
`NeuralBandModel.plot_walkers`.

## Model state documented here

Heading torque is the **half-angle law** `dθ/dt = K·R·sin(ego/2)`.
`plot_walkers` also has a blind-spot search (`blind_search_std`, 
default π/2) that turns a walker with no visible target
into a diffusive search until a target re-enters view.

## Contents

- `make_figures.py` — regenerates the representative figures below.

Run:

```
python walker_analysis/make_figures.py
```

Each figure is a 30-walker heatmap with overlaid tracks. Shared setup: observer
starts at (0,0) facing +x (east); cutoff warp `a=0, b=π` with **uniform**
weighting (`angle_weight=None`); `std=0.1`, `v=1`, `dt=0.1`; seed 0;
`max_steps=1500`. Targets are drawn as red stars. Two target layouts are used,
both at `x=4.33`: **FOUR** deltas at `y=±0.75, ±2.25` and **TWO** at `y=±2.5`;
circle variants give each target radius `r=0.5`.

The figures sweep the turning gain `K` to show how it shapes the approach. 

The K=4 and K=6 figures use a clamped view (`xlim=(-1,7)`, `ylim=(-5,5)`) so a
single wide track does not blow up the auto-scaled axes; the heatmap bins are
still computed over the full range, so in-window resolution is unchanged.

**K-series summary.** `K` sets how hard the walker turns toward its perceived
consensus direction. `K=2` (the default) turns in smoothly; larger `K` turns in
more sharply and can take wider curved excursions on the way, most visibly at
`K=10` with zero-radius (delta) targets. Across the whole sweep no walkers are
lost — finite-radius (circle) targets are, if anything, even more forgiving
because their capture radius absorbs near-misses.
