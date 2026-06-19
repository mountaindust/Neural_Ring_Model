# `plots/` — publication figures

Scripts that produce the publication panels for the fly/locust decision experiments.
Two stages: **simulate + fit** (`*_target_fly_refine.py`), then **plot**
(`fly_results.py`). `decision_skeleton.py` is a separate, deterministic figure.

The model parameterization is **locked to the empirical data** and lives in
`three_target_fly_refine.py` (the two-target and skeleton scripts import or mirror it —
single source of truth). See `three_target_fly_refine_findings.md` /
`two_target_fly_refine_findings.md` for the fit rationale.

## Empirical (GODM) data — where it is expected

The refine scripts score the walker ensemble against the empirical GODM occupancy
heatmaps, computed on the fly by `walker_analysis/godm_heatmaps.compute_heatmap(case)`.
That engine reads the raw GODM experiment data from:

    $GODM_DIR            (default: ../../GODM, i.e. a GODM/ directory beside the repo root)
      Data/flies/<exp_id>/results.csv
      Data/locusts/<exp_id>/results.csv

Override the location with the `GODM_DIR` environment variable. If the data is absent
the refine scripts cannot run (they need it for the reference heatmap + correlation
score); `decision_skeleton.py` degrades gracefully to drawing target circles only.

---

## `three_target_fly_refine.py` / `two_target_fly_refine.py` — simulate + fit

The compute scripts. For a given target geometry they:

1. Build the NBM with the locked knobs, run a large walker ensemble
   (`multiprocessing`, worker count from `parallel_config`).
2. Render the ensemble through the **exact** GODM max-projection pipeline
   (sliding time-window 2-D histograms, per-window normalized, blurred, max-projected)
   so the walker panel is pixel-comparable to the empirical heatmap.
3. Score the match: Pearson `corr(all)` and `corr(support)` vs the empirical heatmap.
4. Write outputs:
   - **`<case>_target_fly_refine.npz`** — self-contained: walker tracks + empirical
     heatmap + every parameter + the correlation. This is the hand-off to the plotter.
   - **`<case>_target_fly_refine.png`** — a 4-panel diagnostic (empirical / walker
     density / overlay+ridge-contour / raw tracks).

`two_target_fly_refine.py` imports its parameterization verbatim from the 3-target
script — same fly, same setup, two targets, no re-tuning — only the geometry and the
GODM case (`fly2` vs `fly3`) differ.

```
python plots/three_target_fly_refine.py            # default reps (1500)
python plots/three_target_fly_refine.py 800        # reps as positional arg
NR_REPS=3000 python plots/three_target_fly_refine.py
python plots/two_target_fly_refine.py              # 2-target sibling, same knobs
```

Environment overrides (all optional; the defaults are the locked fit):
`NR_REPS` (realizations), `NR_K`, `NR_STD`, `NR_A_WARP`, `NR_A_WEIGHT`,
`NR_POS_STD`, `NR_HEAD_STD` (start-jitter sigmas).

## `fly_results.py` — the publication plotter

Loads a `*_target_fly_refine.npz` and renders a single, undistorted, journal-ready
panel: the empirical heatmap with a random subset of walker trajectories overlaid.
**No simulation, no pandas/GODM dependency** — everything is in the npz. Output is
`fly_results_<n>target.png`. Errors out asking you to run the matching refine script
first if the npz is missing.

```
python plots/fly_results.py            # 3-target (default)
python plots/fly_results.py 2          # 2-target  (aliases: 2, 2target, two)
python plots/fly_results.py 3target    # 3-target  (aliases: 3, 3target, three)
NR_MAX_TRACKS=200 python plots/fly_results.py 2    # overlay density (default 100)
```

## `decision_skeleton.py` — deterministic decision-track skeleton

Independent figure: a forking-streamline integration of the model's stable
consensus-heading field (read from `NBM.sc_equilib`), overlaid on the empirical GODM
heatmap (graceful fallback to target circles). K-independent, noise-free — it mirrors
the refine scripts' deterministic knobs (geometry / warp / weight / T) only. Cases:
`fly`, `fly2`, `locust`, `locust2`.

```
python plots/decision_skeleton.py fly                  # skeleton over the fly3 heatmap
python plots/decision_skeleton.py fly --branch-diagram  # (x,theta)+(x,R) bifurcation diagram
python plots/decision_skeleton.py fly2 --no-heatmap     # skeleton over target circles only
python plots/decision_skeleton.py diagram-both          # combined fly+locust heading diagram
```

Useful flags: `--branch-diagram` (Phase-0 bifurcation branch diagram instead of the
skeleton), `--no-heatmap` (target circles only), `--show-unstable` (draw the centre
track through its SC-unstable interlude), `--ds` (streamline step), `--num-x`
(branch-diagram x samples), `--save PATH`, `--no-show`.

Skeleton figures are written here in `plots/`; the analysis-only branch diagrams are
written to `../walker_analysis/`.

## `fly_bifurcation_plot.py` — two-panel bifurcation + basin overlay

Two-panel publication figure: the two-target (left) and three-target (right) fly
stable-equilibrium-count maps, each with the basin-of-attraction wheel overlay
(`NBM.plot_bifurcation_diagram(overlay_basins=True)`). Same fly in both panels; one
shared legend. Writes `fly_bifurcation.jpg` and `.tif` (300 dpi).

```
python plots/fly_bifurcation_plot.py
FLYBIF_FAST=1 python plots/fly_bifurcation_plot.py   # coarse, fast layout check
```

## `combined_walker_figure.py` — publication montage

Composites the finished 300-dpi panels (the four `skeleton_*.png` plus the two
`fly_results_*.png`) into one labelled 2x3 figure (panel letters A-F + titles). Run it
*after* those panel PNGs exist. Writes `combined_walker_figure.jpg` and `.tif`.

```
python plots/combined_walker_figure.py
```

## `combined_branch_figure.py` — combined branch diagram

Reuses `decision_skeleton.plot_diagram_both` to stack the fly SC-equilibrium
*heading* branch row over the locust one (three y-cuts as columns; the per-case
(x, R) coherence rows are dropped), and saves it as a 300-dpi publication pair
`branch_diagram_combined.{jpg,tif}`.

```
python plots/combined_branch_figure.py
```

## Outputs at a glance

| File | Produced by | What it is |
|---|---|---|
| `fly_results_{2,3}target.png` | `fly_results.py` | publication walker panels |
| `{two,three}_target_fly_refine.npz` | refine scripts | tracks + heatmap + params (hand-off) |
| `{two,three}_target_fly_refine.png` | refine scripts | 4-panel fit diagnostic |
| `skeleton_{fly,fly2,locust,locust2}*.png` | `decision_skeleton.py` | deterministic skeletons |
| `fly_bifurcation.{jpg,tif}` | `fly_bifurcation_plot.py` | 2-panel bifurcation + basin overlay |
| `combined_walker_figure.{jpg,tif}` | `combined_walker_figure.py` | publication montage of the six panels |
| `branch_diagram_combined.{jpg,tif}` | `combined_branch_figure.py` | combined fly+locust heading branch diagram |
