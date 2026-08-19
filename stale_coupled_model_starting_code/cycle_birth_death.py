"""Trace the FULL-system limit cycle across the island band (transverse cut):
amplitude, period, and the unstable focus's Re(complex pair). This establishes
the cycle is born/dies via a SUPERCRITICAL HOPF: focus Re crosses 0 at the band
edges with amplitude -> 0 and FINITE period (a homoclinic/SNIC would instead
send the period -> infinity). Measured crossing: Re -0.011 -> +0.020 -> -0.021,
amplitude 0 -> 2.4deg -> 0, period ~12.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import root
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')
from decision_model import Targets, PerceptionModel, NeuralBandModel
from parallel_config import get_n_workers

t = Targets(locs=np.array([[4.33, 2.5], [4.33, -2.5]]), geom_name='circle', r=0.5)


def _model():
    pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='vonmises',
                         angle_weight='neural_angle_dist', a_warp=0.55)
    return NeuralBandModel(pm)


def _probe(args):
    x, y = args
    nbm = _model(); K = nbm.K
    fl = np.array([x, y])
    angs, stab = nbm.sc_equilib(focal_loc=fl)
    n_stable = int(sum(stab)) if len(stab) else 0

    # focus Re(complex pair) of the SC equilibrium
    reC = np.nan
    if len(angs):
        sol = root(nbm._self_consistent_eq, [angs[0], 0.5], args=(fl,),
                   method='hybr', tol=1e-12)
        ts, Rs = sol.x
        J = nbm._coupled_jacobian(Rs + 0j, ts, fl)
        eJ = np.linalg.eigvals(J)
        cp = eJ[np.abs(eJ.imag) > 1e-6]
        if cp.size:
            reC = float(cp.real.max())

    def rhs(tt, s):
        g = s[0] + 1j * s[1]
        dg = nbm.dgamma_dt(gamma=g, focal_angle=s[2], focal_loc=fl)
        return [dg.real, dg.imag, K * np.abs(g) * np.sin(np.angle(g) / 2)]

    # settle, then sample
    s0 = solve_ivp(rhs, (0, 2500), [0.5, 0.02, angs[0] if len(angs) else 0.0],
                   method='LSODA', rtol=1e-9, atol=1e-11)
    sol = solve_ivp(rhs, (0, 1200), s0.y[:, -1], method='LSODA',
                    rtol=1e-10, atol=1e-12, t_eval=np.linspace(0, 1200, 24000))
    th = sol.y[2]; tt = sol.t
    amp = 0.5 * (th[12000:].max() - th[12000:].min())
    pk, _ = find_peaks(th[12000:], prominence=max(1e-5, 0.05 * amp))
    period = np.mean(np.diff(tt[12000:][pk])) if len(pk) >= 2 else np.nan
    return x, y, n_stable, reC, np.degrees(amp), period


if __name__ == '__main__':
    # parametrize the arc by x, with y tracking the band center
    xs = np.linspace(1.66, 2.52, 30)
    ys = 2.07 + 0.62 * (xs - 1.66) / (2.52 - 1.66)   # rough band centerline
    pts = list(zip([float(x) for x in xs], [float(y) for y in ys]))
    with Pool(get_n_workers()) as pool:
        res = pool.map(_probe, pts)
    print(f"{'x':>6} {'y':>6} {'nstab':>5} {'focusRe':>9} {'amp(deg)':>9} {'period':>8}")
    for x, y, ns, reC, amp, per in res:
        print(f"{x:6.3f} {y:6.3f} {ns:5d} {reC:9.4f} {amp:9.2f} "
              f"{per if not np.isnan(per) else float('nan'):8.2f}")
    # edges of the cycle region
    cyc = [(x, y, amp, per) for x, y, ns, reC, amp, per in res if amp > 0.5]
    if cyc:
        print(f"\ncycle present at {len(cyc)}/{len(res)} probed points")
        lo, hi = cyc[0], cyc[-1]
        print(f"  low-x edge:  ({lo[0]:.3f},{lo[1]:.3f}) amp={lo[2]:.2f} period={lo[3]:.2f}")
        print(f"  high-x edge: ({hi[0]:.3f},{hi[1]:.3f}) amp={hi[2]:.2f} period={hi[3]:.2f}")
