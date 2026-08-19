# Worker-count configuration

The number of multiprocessing workers used by parallel scripts in this
project is set per-machine via [parallel_config.py](parallel_config.py).
Every `.py` site that creates a `multiprocessing.Pool` resolves its worker
count through `parallel_config.get_n_workers()`. To change the default on a
new machine, edit [machine_config.py](machine_config.py) (or copy
[machine_config.template.py](machine_config.template.py) to that name on a
fresh checkout).

This file used to log a temporary `10 → 4` downshift for a 4-core Windows
laptop; that ad-hoc approach has been replaced by the config-based
mechanism described below. The historical change set is in the git log if
needed.

## How it works

`parallel_config.get_n_workers()` resolves the worker count in this order:

1. `NR_N_WORKERS` environment variable, if set. Useful for a single run or
   a cluster job script:
   ```powershell
   $env:NR_N_WORKERS = "12"
   python plots/neural_weight_sweep.py
   ```
2. `N_WORKERS` from [machine_config.py](machine_config.py), if present.
   This file is gitignored so each machine keeps its own copy.
3. A built-in default (currently `4`) if neither of the above is set.
   This lets a fresh clone run without any setup.

## Switching machines

On a new machine, copy the template:

```powershell
Copy-Item machine_config.template.py machine_config.py
```

and edit `N_WORKERS` to suit. Typical values:

| Machine                  | `N_WORKERS` |
| ---                      | ---:        |
| 4-core Windows-10 laptop | 4           |
| 12-core workstation      | 10          |

(Leave one or two cores free on machines you also use interactively.)

## Call sites

These are the `.py` files that read `get_n_workers()`. They are listed so
that anyone adding a new parallel script knows where to look for the
existing pattern. To run any one of them with a different worker count
without editing files, set `NR_N_WORKERS` in the shell.

| File | Symbol / call |
| --- | --- |
| [plots/neural_weight_sweep.py](plots/neural_weight_sweep.py) | `DEFAULT_N_WORKERS = get_n_workers()` (also `--workers` CLI override) |
| [plots/fly_bifurcation_plot.py](plots/fly_bifurcation_plot.py) | `with Pool(get_n_workers()) as pool:` |
| [plots/fly_geom.py](plots/fly_geom.py) | `with Pool(get_n_workers()) as pool:` |
| [plots/horn_decision_figure.py](plots/horn_decision_figure.py) | `with Pool(get_n_workers(), initializer=_init_worker) as pool:` |
| [plots/oblique_walker.py](plots/oblique_walker.py) | `with Pool(get_n_workers()) as pool:` |
| [plots/oblique_walker_uniform_check.py](plots/oblique_walker_uniform_check.py) | `with ow.Pool(ow.get_n_workers()) as pool:` |
| [plots/three_target_fly_refine.py](plots/three_target_fly_refine.py) | `n_workers = get_n_workers()` |
| [plots/two_target_fly_refine.py](plots/two_target_fly_refine.py) | `n_workers = get_n_workers()` |
| [archive/stale_coupled_model_starting_code/compare_reduced_vs_coupled.py](archive/stale_coupled_model_starting_code/compare_reduced_vs_coupled.py) (stale) | `with Pool(get_n_workers()) as pool:` |
| [archive/stale_coupled_model_starting_code/cycle_birth_death.py](archive/stale_coupled_model_starting_code/cycle_birth_death.py) | `with Pool(get_n_workers()) as pool:` |
| [archive/stale_coupled_model_starting_code/island_anatomy.py](archive/stale_coupled_model_starting_code/island_anatomy.py) | `with Pool(get_n_workers()) as pool:` |
| [archive/stale_coupled_model_starting_code/reduced_vs_full_dynamics.py](archive/stale_coupled_model_starting_code/reduced_vs_full_dynamics.py) | `with Pool(get_n_workers()) as pool:` |
| [weighting_analysis/ears_figure.py](weighting_analysis/ears_figure.py) | `N_WORKERS = get_n_workers()` |
| [weighting_analysis/outward_bias.py](weighting_analysis/outward_bias.py) | `N_WORKERS = get_n_workers()` |
| [tests/test_broad_validation.py](tests/test_broad_validation.py) | `with Pool(get_n_workers()) as pool:` |

Note: [decision_model.py](decision_model.py) does **not** create its own
pool; several of its methods accept a `pool=` keyword argument and the
caller (one of the scripts above) is responsible for sizing it.

## Notebooks

The following notebooks still embed hard-coded `Pool(...)` calls and were
not migrated to `get_n_workers()` in this pass (notebook diffs are noisy
and the day-to-day flow runs the `.py` scripts). Each is annotated with
its current literal value; convert opportunistically when next editing.

| File | Line | Current value |
| --- | ---: | --- |
| [compare_sc_beta.ipynb](compare_sc_beta.ipynb) | 75 | `Pool(10)` |
| [compare_sc_vm.ipynb](compare_sc_vm.ipynb) | 73 | `Pool(10)` |
| [debug_all_unstable.ipynb](debug_all_unstable.ipynb) | 50 | `Pool(10)` |
| [debug_all_unstable.ipynb](debug_all_unstable.ipynb) | 76 | `Pool(10)` |
| [neural_band.ipynb](neural_band.ipynb) | 68 | `Pool(10)` |
| [neural_band.ipynb](neural_band.ipynb) | 133 | `Pool(10)` |
| [neural_band.ipynb](neural_band.ipynb) | 147 | `Pool(10)` |
| [neural_band.ipynb](neural_band.ipynb) | 210 | `Pool(10)` |
| [neural_band.ipynb](neural_band.ipynb) | 272 | `Pool(10)` |
| [neural_band.ipynb](neural_band.ipynb) | 334 | `Pool(10)` |
| [ising_workbook.ipynb](ising_workbook.ipynb) | 56 | `Pool(10)` |

To migrate a notebook cell, replace `Pool(N)` with `Pool(get_n_workers())`
and add `from parallel_config import get_n_workers` to an early setup
cell. (Notebooks import from the repo root via the usual `sys.path`
adjustment they already do.)
