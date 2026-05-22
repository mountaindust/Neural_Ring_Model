"""Machine-specific defaults. Copy this file to ``machine_config.py`` and
edit ``N_WORKERS`` to suit the current machine.

``machine_config.py`` is gitignored so each machine keeps its own copy.
See ``parallel_config.get_n_workers`` for the resolution order.
"""

# Size of the default multiprocessing pool on this machine.
# Pick a value that leaves the machine usable while you work
# (e.g. cores - 1 on a dedicated workstation, fewer on a laptop).
N_WORKERS = 4
