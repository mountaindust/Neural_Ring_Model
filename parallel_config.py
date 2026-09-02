# Neural Ring Model: Ising-type dynamics of spatial decision-making.
# Copyright (C) 2026 Christopher Strickland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Machine-specific default worker count for multiprocessing pools.

Resolves the worker count for `multiprocessing.Pool(...)` call sites in this
project. The number of cores varies between machines (4-core laptop,
12-core workstation, etc.); rather than hard-coding a value in every script,
each script calls ``get_n_workers()``.

Resolution order (highest priority first):

    1. ``NR_N_WORKERS`` environment variable, if set.
       (One-off override; useful for a single run or a cluster job script.)
    2. ``N_WORKERS`` from a top-level ``machine_config.py`` module, if
       present. ``machine_config.py`` is NOT committed -- each machine has
       its own copy. ``machine_config.template.py`` is the starting point.
    3. The ``default`` argument (4 if not specified).

Typical usage at a call site::

    from parallel_config import get_n_workers
    N_WORKERS = get_n_workers()
    ...
    with Pool(N_WORKERS) as pool:
        ...
"""

import os


def get_n_workers(default: int = 4) -> int:
    env = os.environ.get("NR_N_WORKERS")
    if env is not None:
        return int(env)
    try:
        from machine_config import N_WORKERS
        return int(N_WORKERS)
    except ImportError:
        return default
