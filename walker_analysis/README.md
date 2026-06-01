# walker_analysis

Walker (physical-space, x-y) random-walk analysis for `NeuralBandModel`.

This is distinct from [`basin_estimation/`](../basin_estimation/), which deals
with noise in the **coherence** variable γ (γ-Langevin / Kramers escape for
basin geometry). Here the noise is angular heading noise and the object of
study is the walker's trajectory in physical space via
`NeuralBandModel.plot_walkers`.

## Model state documented here

Heading torque is the **half-angle law** `dθ/dt = K·R·sin(ego/2)` with default
`K=2` (see the "Half-angle heading torque" section of the top-level
[CLAUDE.md](../CLAUDE.md)). `plot_walkers` also has a blind-spot search
(`blind_search_std`, default π/2) that turns a walker with no visible target
into a diffusive search until a target re-enters view.

## Contents

- `make_figures.py` — regenerates the five representative figures below.
- `compare.py` — before/after harness used to validate the change
  (`before` / `after` / `diff` modes): lost-walker counts + a bifurcation
  stable-count raster. Produced the 8/30 → 2/30 four-delta result and the
  bit-identical raster confirming the K-doubling invariance.
- `four_delta_{before,after}.png`, `two_circle_{before,after}.png` —
  the before/after heatmaps from `compare.py`.

## Representative figures (`make_figures.py`)

Observer starts at (0,0) facing +x; cutoff warp `a=0, b=π` with full (tied)
weighting; `std=0.5`, `v=1`, `dt=0.1`, 30 walkers. Titles carry the K value.

| figure | targets | K | note |
|--------|---------|---|------|
| `walkers_4delta_K1.png`  | 4 delta  | 1  | 2/30 *hover* 0.4–0.8 from a target (under-turns, doesn't quite land) |
| `walkers_4circle_K1.png` | 4 circle (r=0.5) | 1 | 0/30 lost, clean homing |
| `walkers_4delta_K2.png`  | 4 delta  | 2  | clean homing |
| `walkers_4circle_K2.png` | 4 circle (r=0.5) | 2 | clean homing |
| `walkers_4delta_K4.png`  | 4 delta  | 4  | 0/30 lost; wide orbit *excursions* before landing |
| `walkers_4circle_K4.png` | 4 circle (r=0.5) | 4 | 0/30 lost, clean homing |
| `walkers_4delta_K10.png` | 4 delta  | 10 | strong-coupling **wide orbits** — an artifact of large K + zero-radius point targets, not a perception failure |
| `walkers_2delta_K1.png`  | 2 delta  | 1  | 5/30 **loop around the pair** without committing (under-turns) |
| `walkers_2circle_K1.png` | 2 circle (r=0.5) | 1 | 0/30 lost, clean homing |
| `walkers_2delta_K2.png`  | 2 delta  | 2  | clean homing |
| `walkers_2circle_K2.png` | 2 circle (r=0.5) | 2 | clean homing |
| `walkers_2delta_K4.png`  | 2 delta  | 4  | 1/30 escapes (delta + intermediate K); one track exits the frame |
| `walkers_2circle_K4.png` | 2 circle (r=0.5) | 4 | 0/30 lost, clean homing |

The K=1 and K=4 figures use a clamped view (`xlim=(-1,7)`, `ylim=(-5,5)`) so a
single escaping/orbiting track does not blow up the auto-scaled axes; the
heatmap bins are still computed over the full range, so in-window resolution is
unchanged.

**K-series summary (zero-radius/delta targets are the stress case).** Both
extremes cause orbiting around point targets — K=1 *under*-turns and loops
around the cluster/pair without landing; K≥4 *over*-turns into wide orbits, and
K=10 strongly so — while **K=2 is the clean middle** (the default). Finite-radius
(circle) targets lose **no** walkers at any K tested, because their capture
radius absorbs the near-miss hoverers. Note K=1 here is *gentler* than the old
`sin(ego)` default: `sin(ego/2)` already halves the near-front gain, so old
`K=1, sin(ego)` ≈ new `K=2, sin(ego/2)`.
