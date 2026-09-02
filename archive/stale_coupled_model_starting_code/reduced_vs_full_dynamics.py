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

"""Reduced (slaved) dynamics vs full coupled dynamics in the 0-stable band.

FULL system: integrate the 3D ODE (gamma_re, gamma_im, theta) together, gamma a
   live variable on the theta timescale (epsilon = 1).
REDUCED (slaved) system: the deterministic core of plot_walkers -- at each
   theta-step run gamma to equilibrium (warm-started, so it tracks a branch),
   then step theta by dt * K*R*sin(Theta/2). gamma is ALWAYS at equilibrium;
   it never lags. This is a 1-D flow on theta.

The head-bobbing limit cycle is a FULL-system object (it needs gamma to lag).
The reduced/slaved walker does not reproduce it: at this point gamma is bistable,
so the slaved walker instead executes a much smaller, faster relaxation
oscillation across the bistable window (see reduced_dynamics_anatomy.py).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')
from decision_model import Targets, PerceptionModel, NeuralBandModel, convert_angles
from parallel_config import get_n_workers

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FL = (2.1, 2.45)


def _model():
    t = Targets(locs=np.array([[4.33, 2.5], [4.33, -2.5]]), geom_name='circle', r=0.5)
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='vonmises',
                         angle_weight='neural_angle_dist', a_warp=0.55)
    return NeuralBandModel(pm)


def run_full(args):
    theta0, = args
    nbm = _model(); K = nbm.K; fl = np.array(FL)

    def rhs(tt, s):
        g = s[0] + 1j * s[1]
        dg = nbm.dgamma_dt(gamma=g, focal_angle=s[2], focal_loc=fl)
        return [dg.real, dg.imag, K * np.abs(g) * np.sin(np.angle(g) / 2)]

    sol = solve_ivp(rhs, (0, 600), [0.05, 0.02, theta0], method='LSODA',
                    rtol=1e-9, atol=1e-11, t_eval=np.linspace(0, 600, 6000))
    return ('full', theta0, sol.t, sol.y[2], np.hypot(sol.y[0], sol.y[1]))


def run_reduced(args):
    """Slaved walker: gamma fully relaxed to equilibrium each theta-step,
    warm-started from the previous gamma so it follows a continuous branch.
    Warm-started gamma converges fast, so a short relaxation suffices."""
    theta0, = args
    nbm = _model(); K = nbm.K; fl = np.array(FL)
    dt = 0.1; tF = 600
    nsteps = int(tF / dt)
    theta = theta0
    gamma = 0.05 + 0.02j
    ts = np.empty(nsteps); ths = np.empty(nsteps); Rs = np.empty(nsteps)
    for k in range(nsteps):
        gamma = nbm.run_dgamma_dt(focal_angle=theta, focal_loc=fl,
                                  init_gamma=gamma, t_Final=40)
        R = np.abs(gamma); Theta = np.angle(gamma)
        theta = convert_angles(theta + K * R * np.sin(Theta / 2) * dt)
        ts[k] = k * dt; ths[k] = theta; Rs[k] = R
    return ('reduced', theta0, ts, ths, Rs)


if __name__ == '__main__':
    nbm = _model()
    angs, stab = nbm.sc_equilib(focal_loc=np.array(FL))
    print(f"focal_loc {FL}: SC equilibria headings={np.round(angs,4)}, "
          f"n_stable={sum(stab)}")

    th0s = [-0.8, -0.0876, 0.6]
    jobs = [(th0,) for th0 in th0s]
    with Pool(get_n_workers()) as pool:
        full_res = pool.map(run_full, jobs)
        red_res = pool.map(run_reduced, jobs)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for _, th0, t, th, R in full_res:
        ax[0, 0].plot(t, th, lw=1, label=f'θ₀={th0:+.2f}')
        ax[0, 1].plot(t, R, lw=1)
    for _, th0, t, th, R in red_res:
        ax[1, 0].plot(t, th, lw=1, label=f'θ₀={th0:+.2f}')
        ax[1, 1].plot(t, R, lw=1)
    ax[0, 0].set_title('FULL coupled: θ(t)  (γ live, ε=1)')
    ax[0, 1].set_title('FULL coupled: R(t)')
    ax[1, 0].set_title('REDUCED slaved: θ(t)  (γ at equilibrium each step)')
    ax[1, 1].set_title('REDUCED slaved: R(t)')
    for a in ax[:, 0]:
        a.set_xlabel('t'); a.set_ylabel('θ'); a.legend(fontsize=8)
        for th in angs:
            a.axhline(th, color='r', ls='--', lw=0.7)
    for a in ax[:, 1]:
        a.set_xlabel('t'); a.set_ylabel('R = |γ|')
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'reduced_vs_full.png')
    fig.savefig(out, dpi=110)
    print("saved", out)

    print("\nθ-oscillation amplitude (deg), last half of trajectory:")
    fd = {th0: (th, R) for _, th0, t, th, R in full_res}
    rd = {th0: (th, R) for _, th0, t, th, R in red_res}
    for th0 in th0s:
        thf, _ = fd[th0]; thr, _ = rd[th0]
        af = np.degrees(0.5 * (thf[len(thf)//2:].max() - thf[len(thf)//2:].min()))
        ar = np.degrees(0.5 * (thr[len(thr)//2:].max() - thr[len(thr)//2:].min()))
        print(f"  θ₀={th0:+.3f}:  FULL amp={af:6.2f}   REDUCED amp={ar:6.2f}"
              f"   (reduced θ_final={thr[-1]:+.4f}, full θ_mean={thf[len(thf)//2:].mean():+.4f})")
