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

"""Anatomy of the vonmises island region: what is the 0-stable region, where is
the (weak) Hopf, and how does the reduced criterion behave as eig(A) -> 0
(near a gamma-fold, where the Schur inverse A^{-1} is ill-conditioned)?

For each observer (x,y) and each SC equilibrium we classify:
  - 'saddle'   : A NOT Hurwitz (fast/gamma-unstable). reduced=coupled=unstable.
  - 'slow_unst': A Hurwitz but Schur lam_slow > 0 (slow theta-unstable).
  - 'hopf'     : A Hurwitz, lam_slow < 0 (reduced-stable), yet coupled has a
                 COMPLEX eigenvalue pair with positive real part (Hopf focus).
  - 'stable'   : reduced-stable AND coupled-stable.
and record min|eig(A)| (closeness to a fold) and |lam_slow| (Schur blow-up).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.optimize import root
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')
from decision_model import Targets, PerceptionModel, NeuralBandModel
from parallel_config import get_n_workers

OUTDIR = os.path.dirname(os.path.abspath(__file__))
TWO_CIRCLE = np.array([[4.33, 2.5], [4.33, -2.5]])
TOL = 1e-8


def _model():
    t = Targets(locs=TWO_CIRCLE, geom_name='circle', r=0.5)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='vonmises',
                         angle_weight='neural_angle_dist', a_warp=0.55)
    return NeuralBandModel(pm)


def _schur(J):
    A, b, c, d = J[:2, :2], J[:2, 2], J[2, :2], J[2, 2]
    return d - c @ np.linalg.solve(A, b)


def classify(nbm, fl):
    angs, _ = nbm.sc_equilib(focal_loc=fl)
    n_red = n_cpl = 0
    min_absA = np.inf
    max_lam = 0.0
    hopf = -np.inf
    kinds = []
    for th in angs:
        sol = root(nbm._self_consistent_eq, [th, 0.5], args=(fl,),
                   method='hybr', tol=1e-12)
        ts, Rs = sol.x
        if Rs < 0.05:
            continue
        J = nbm._coupled_jacobian(Rs + 0j, ts, fl)
        eA = np.linalg.eigvals(J[:2, :2]).real
        eJ = np.linalg.eigvals(J)
        min_absA = min(min_absA, float(np.min(np.abs(eA))))
        A_hur = np.all(eA < -TOL)
        red = bool(A_hur and _schur(J) < -TOL)
        cpl = bool(np.all(eJ.real < -TOL))
        n_red += int(red)
        n_cpl += int(cpl)
        if not A_hur:
            kinds.append('saddle')
        else:
            lam = float(_schur(J))
            max_lam = max(max_lam, abs(lam))
            cplx = eJ[np.abs(eJ.imag) > 1e-6]
            if lam >= -TOL:
                kinds.append('slow_unst')
            elif cplx.size and cplx.real.max() > 1e-6:
                hopf = max(hopf, float(cplx.real.max()))
                kinds.append('hopf')
            else:
                kinds.append('stable')
    return n_red, n_cpl, min_absA, max_lam, hopf, kinds


def _worker(args):
    x, y = args
    nbm = _model()
    n_red, n_cpl, min_absA, max_lam, hopf, kinds = classify(nbm, np.array([x, y]))
    return (x, y, n_red, n_cpl, min_absA, max_lam, hopf, ';'.join(kinds))


if __name__ == '__main__':
    xs = np.linspace(1.7, 2.9, 73)   # ~0.0167 spacing; lands on the Hopf cell
    ys = np.linspace(2.0, 3.0, 61)   # ~0.0167 spacing, upper island only
    args = [(float(x), float(y)) for y in ys for x in xs]
    with Pool(get_n_workers()) as pool:
        res = pool.map(_worker, args)

    nx, ny = len(xs), len(ys)
    red = np.zeros((ny, nx), int); cpl = np.zeros((ny, nx), int)
    hopf = np.full((ny, nx), -np.inf); minA = np.zeros((ny, nx))
    for x, y, nr, nc, mA, ml, hf, kinds in res:
        i = int(np.argmin(np.abs(xs - x))); j = int(np.argmin(np.abs(ys - y)))
        red[j, i] = nr; cpl[j, i] = nc; hopf[j, i] = hf; minA[j, i] = mA

    # --- text anatomy ---
    print(f"grid {nx}x{ny} over x[1.8,2.9] y[2.0,3.0]")
    print(f"reduced counts: {dict(zip(*np.unique(red, return_counts=True)))}")
    print(f"coupled counts: {dict(zip(*np.unique(cpl, return_counts=True)))}")

    print("\n0-stable cells (reduced): kinds of their equilibria")
    z = [(x, y, kinds) for (x, y, nr, nc, mA, ml, hf, kinds) in res if nr == 0]
    from collections import Counter
    allkinds = Counter()
    for x, y, kinds in z:
        for k in kinds.split(';'):
            allkinds[k] += 1
    print(f"  {len(z)} zero-stable cells; equilibrium kinds: {dict(allkinds)}")
    print(f"  sample: {[(round(x,3),round(y,3),k) for x,y,k in z[:4]]}")

    print("\nHopf cells (A Hurwitz, coupled complex pair Re>1e-6):")
    h = [(x, y, hf) for (x, y, nr, nc, mA, ml, hf, kinds) in res if hf > 1e-6]
    for x, y, hf in h:
        print(f"  ({x:.3f},{y:.3f}) Re={hf:+.5f}")
    if not h:
        print("  none in window")

    print("\nNear-fold cells (min|eig(A)| < 0.03) and Schur magnitude:")
    nf = sorted([(mA, x, y, ml) for (x, y, nr, nc, mA, ml, hf, kinds) in res
                 if mA < 0.03])
    for mA, x, y, ml in nf[:8]:
        print(f"  ({x:.3f},{y:.3f}) min|eig(A)|={mA:.2e}  max|lam_slow|={ml:.3g}")
    print(f"  total near-fold cells: {len(nf)}")

    # --- figure: 4 panels ---
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    vmax = int(max(red.max(), cpl.max(), 1))
    cmap = plt.get_cmap('viridis', vmax + 1)
    norm = BoundaryNorm(np.arange(-0.5, vmax + 1.5, 1), cmap.N)
    for a, data, title in [(ax[0], red, 'reduced (# stable eq)'),
                          (ax[1], cpl, 'coupled (# stable eq)')]:
        im = a.imshow(data, origin='lower', extent=extent, aspect='equal',
                      interpolation='nearest', cmap=cmap, norm=norm)
        a.set_title(title); fig.colorbar(im, ax=a, ticks=range(vmax+1))

    diff = red - cpl
    dm = int(max(abs(diff).max(), 1))
    im2 = ax[2].imshow(diff, origin='lower', extent=extent, aspect='equal',
                       interpolation='nearest', cmap='RdBu_r', vmin=-dm, vmax=dm)
    ax[2].set_title('reduced - coupled'); fig.colorbar(im2, ax=ax[2])

    hh = np.where(np.isfinite(hopf), hopf, np.nan)
    im3 = ax[3].imshow(hh, origin='lower', extent=extent, aspect='equal',
                       interpolation='nearest', cmap='inferno')
    ax[3].set_title('Hopf focus Re(eig) (coupled, A Hurwitz)')
    fig.colorbar(im3, ax=ax[3])
    # mark cells where coupled actually has a Hopf-unstable focus
    for x, y, hf in h:
        for a in ax:
            a.plot(x, y, 'r+', ms=12, mew=2)
    for a in ax:
        a.set_xlabel('x'); a.set_ylabel('y')
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'island_anatomy.png')
    fig.savefig(out, dpi=110)
    print("\nsaved", out)
