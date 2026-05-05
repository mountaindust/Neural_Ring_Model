# Hardware-temporary worker-count downshift

This is a temporary log of all `.py` sites where the parallel-pool worker count
was lowered from `10` to `4` to suit a 4-core Windows-10 laptop. Restore each
line back to `10` once back on the 12-core workstation.

Every changed line is tagged with the comment:

```
# HW-TEMP: 4-core laptop; restore to 10 on main workstation
```

so the full change set can be rediscovered with:

```
grep -rn HW-TEMP .
```

## Sites

| File | Line | Original | Replaced with |
| --- | ---: | --- | --- |
| [publication_plots/neural_weight_sweep.py](publication_plots/neural_weight_sweep.py) | 73 | `N_WORKERS = 10` | `N_WORKERS = 4` |
| [publication_plots/arc_skeleton_and_island_dynamics.py](publication_plots/arc_skeleton_and_island_dynamics.py) | 75 | `N_WORKERS = 10` | `N_WORKERS = 4` |
| [publication_plots/bifurcation_compare_discrim_vs_coupled.py](publication_plots/bifurcation_compare_discrim_vs_coupled.py) | 61 | `N_WORKERS = 10` | `N_WORKERS = 4` |
| [plot_dir_mesh.py](plot_dir_mesh.py) | 52 | `with Pool(10) as pool:` | `with Pool(4) as pool:` |
| [test_broad_validation.py](test_broad_validation.py) | 143 | `with Pool(10) as pool:` | `with Pool(4) as pool:` |
| [VM_bifurcations/diagnostic_arc_skeleton.py](VM_bifurcations/diagnostic_arc_skeleton.py) | 139 | `with Pool(10) as pool:` | `with Pool(4) as pool:` |
| [VM_bifurcations/diagnostic_arc_bifurcation.py](VM_bifurcations/diagnostic_arc_bifurcation.py) | 215 | `with Pool(10) as pool:` | `with Pool(4) as pool:` |
| [VM_bifurcations/diagnostic_island_final.py](VM_bifurcations/diagnostic_island_final.py) | 109 | `with Pool(10) as pool:` | `with Pool(4) as pool:` |
| [VM_bifurcations/diagnostic_recount_grid.py](VM_bifurcations/diagnostic_recount_grid.py) | 122 | `with Pool(10) as pool:` | `with Pool(4) as pool:` |

(Line numbers reflect the position immediately *after* the comment line was
inserted; the underlying code line is one below the `HW-TEMP` comment.)

## Out of scope

The following `.ipynb` notebooks also embed `with Pool(10) as pool:` cells but
were intentionally left unchanged in this round (notebook diffs are noisy and
the user's day-to-day flow runs the `.py` scripts):

- `debug_all_unstable.ipynb`
- `compare_sc_vm.ipynb`
- `compare_sc_beta.ipynb`
- `neural_band.ipynb` (5 cells)
- `ising_workbook.ipynb`

If a notebook run becomes painful on the 4-core box, lower its `Pool(10)` cell
manually before running and revert before committing.

## Reverting

```
grep -rln HW-TEMP . | xargs sed -i 's/Pool(4)/Pool(10)/g; s/N_WORKERS = 4/N_WORKERS = 10/g'
grep -rln HW-TEMP . | xargs sed -i '/HW-TEMP: 4-core laptop/d'
```

Then `grep -rn HW-TEMP .` should produce no output. (On Windows, run these
sed lines from Git Bash or WSL; PowerShell's `Set-Content` form would
require slight rewording.)
