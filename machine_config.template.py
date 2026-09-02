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

"""Machine-specific defaults. Copy this file to ``machine_config.py`` and
edit ``N_WORKERS`` to suit the current machine.

``machine_config.py`` is gitignored so each machine keeps its own copy.
See ``parallel_config.get_n_workers`` for the resolution order.
"""

# Size of the default multiprocessing pool on this machine.
# Pick a value that leaves the machine usable while you work
# (e.g. cores - 1 on a dedicated workstation, fewer on a laptop).
N_WORKERS = 4
