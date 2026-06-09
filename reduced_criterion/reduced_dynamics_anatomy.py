"""Anatomy of the REDUCED (slaved / uncoupled) dynamics in the 0-stable band.

The fast gamma-subsystem is BISTABLE over a heading window: two stable
gamma-branches (Theta_neur>0 and <0) flank an unstable symmetric branch
(Theta_neur=0, the SC equilibrium). The slaved slow flow dtheta/dt = g(h(theta))
is therefore MULTIVALUED. A branch-tracking slaved walker rides a branch until
it folds, then jumps -> a relaxation oscillation. This script maps the branches,
builds the idealized relaxation cycle (period by integrating dt=dtheta/g along
the branches), and overlays an actual warm-started Euler slaved walker.

Four panels:
  (A) (theta, dtheta/dt): the multivalued slow flow + the relaxation cycle.
  (B) (theta, Theta_neur): the three gamma-branches (hysteresis in the angle).
  (C) theta(t): idealized sawtooth (branch integration) + Euler slaved walker.
  (D) gamma-plane (Re,Im): the branches and the jump (hysteresis in gamma).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from decision_model import Targets, PerceptionModel, NeuralBandModel, convert_angles

OUTDIR = os.path.dirname(os.path.abspath(__file__))
t = Targets(locs=np.array([[4.33, 2.5], [4.33, -2.5]]), geom_name='circle', r=0.5)
pm = PerceptionModel(t, (0, 0), 0, neural_angle_dist='vonmises',
                     angle_weight='neural_angle_dist', a_warp=0.55)
nbm = NeuralBandModel(pm); K = nbm.K
fl = np.array([2.1, 2.45])


def dtheta_of(g):
    return K * np.abs(g) * np.sin(np.angle(g) / 2)


# ---- 1. map all gamma-branches vs heading (stable + unstable) ----
ths = np.linspace(-0.16, 0.02, 181)
up = {'th': [], 'd': [], 'Th': [], 'g': []}     # stable, Theta_neur > 0
lo = {'th': [], 'd': [], 'Th': [], 'g': []}     # stable, Theta_neur < 0
un = {'th': [], 'd': [], 'Th': [], 'g': []}     # unstable (symmetric SC branch)
for th in ths:
    geqs, st = nbm.gamma_equilib(focal_angle=th, focal_loc=fl,
                                 stability_criterion='discrim_a')
    for g, s in zip(geqs, st):
        Th = np.angle(g); d = dtheta_of(g)
        rec = (un if not s else (up if Th >= 0 else lo))
        rec['th'].append(th); rec['d'].append(d); rec['Th'].append(Th); rec['g'].append(g)
for rec in (up, lo, un):
    for k in rec:
        rec[k] = np.array(rec[k])

# window edges (where each stable branch folds)
th_top = up['th'].max() if up['th'].size else np.nan   # upper branch dies (top fold)
th_bot = lo['th'].min() if lo['th'].size else np.nan   # lower branch dies (bottom fold)
print(f"bistable window: theta in [{th_bot:.4f}, {th_top:.4f}]  "
      f"(~{np.degrees(th_top-th_bot):.2f} deg wide)")
print(f"upper branch dies at top fold theta={th_top:.4f}: "
      f"Theta_neur={up['Th'][up['th'].argmax()]:+.4f}, dtheta/dt={up['d'][up['th'].argmax()]:+.4f}")
print(f"lower branch dies at bottom fold theta={th_bot:.4f}: "
      f"Theta_neur={lo['Th'][lo['th'].argmin()]:+.4f}, dtheta/dt={lo['d'][lo['th'].argmin()]:+.4f}")

# ---- 2. idealized relaxation cycle: integrate dt = dtheta / |g| along branches ----
# upper branch within the window, ascending; lower branch within window, descending
def branch_in_window(rec, ascending):
    mask = (rec['th'] >= th_bot - 1e-9) & (rec['th'] <= th_top + 1e-9)
    th = rec['th'][mask]; d = rec['d'][mask]
    order = np.argsort(th) if ascending else np.argsort(th)[::-1]
    return th[order], d[order]

thu, du = branch_in_window(up, ascending=True)     # ride up, du>0
thl, dl = branch_in_window(lo, ascending=False)    # ride down, dl<0

# cumulative time along each leg: dt = dtheta / dtheta_dt
def leg_time(th, d):
    dth = np.abs(np.diff(th))
    speed = np.abs(0.5 * (d[:-1] + d[1:]))
    dt = dth / np.clip(speed, 1e-6, None)
    return np.concatenate([[0], np.cumsum(dt)])

tu = leg_time(thu, du)
tl = leg_time(thl, dl)
T_up, T_lo = tu[-1], tl[-1]
period = T_up + T_lo
print(f"idealized relaxation-oscillation period ~ {period:.3f} "
      f"(up leg {T_up:.3f} + down leg {T_lo:.3f}); amplitude "
      f"{np.degrees(th_top-th_bot):.2f} deg p-p")

# stitch a few periods of theta(t)
seg_t = np.concatenate([tu, T_up + tl])
seg_th = np.concatenate([thu, thl])
nper = 6
ideal_t = np.concatenate([seg_t + k*period for k in range(nper)])
ideal_th = np.tile(seg_th, nper)

# ---- 3. actual warm-started Euler slaved walker ----
def slaved_walker(theta0, dt=0.01, tF=8.0, tfin=80):
    n = int(tF/dt); th = theta0; g = 0.6 + 0.05j
    ts = np.empty(n); ths = np.empty(n); gs = np.empty(n, complex)
    for k in range(n):
        g = nbm.run_dgamma_dt(focal_angle=th, focal_loc=fl, init_gamma=g, t_Final=tfin)
        th = convert_angles(th + dtheta_of(g) * dt)
        ts[k] = k*dt; ths[k] = th; gs[k] = g
    return ts, ths, gs

wt, wth, wg = slaved_walker(-0.08)

# ---- figure ----
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# (A) multivalued slow flow
A = ax[0, 0]
A.axvspan(th_bot, th_top, color='gray', alpha=0.15, label='bistable γ window')
A.axhline(0, color='k', lw=0.6)
A.plot(up['th'], up['d'], '-', color='C0', lw=2, label='stable branch Θ>0  (dθ/dt>0)')
A.plot(lo['th'], lo['d'], '-', color='C3', lw=2, label='stable branch Θ<0  (dθ/dt<0)')
A.plot(un['th'], un['d'], '--', color='gray', lw=1.5, label='unstable symmetric (SC eq)')
# relaxation cycle arrows
A.annotate('', xy=(th_top, du[-1]), xytext=(th_bot, du[0]),
           arrowprops=dict(arrowstyle='-|>', color='C0', lw=2))
A.annotate('', xy=(th_bot, dl[-1]), xytext=(th_top, dl[0]),
           arrowprops=dict(arrowstyle='-|>', color='C3', lw=2))
A.annotate('jump ↓', xy=(th_top, 0), xytext=(th_top+0.005, 0.05), color='k', fontsize=9)
A.annotate('jump ↑', xy=(th_bot, 0), xytext=(th_bot-0.03, -0.06), color='k', fontsize=9)
A.set_xlabel('θ (heading)'); A.set_ylabel('dθ/dt = g(h(θ))')
A.set_title('(A) Reduced slow flow is MULTIVALUED → relaxation cycle')
A.legend(fontsize=8, loc='upper right')

# (B) Theta_neur branches (hysteresis in the order-parameter angle)
B = ax[0, 1]
B.axvspan(th_bot, th_top, color='gray', alpha=0.15)
B.axhline(0, color='k', lw=0.6)
B.plot(up['th'], up['Th'], '-', color='C0', lw=2, label='stable Θ>0')
B.plot(lo['th'], lo['Th'], '-', color='C3', lw=2, label='stable Θ<0')
B.plot(un['th'], un['Th'], '--', color='gray', lw=1.5, label='unstable (Θ=0)')
B.annotate('', xy=(th_top, 0.02), xytext=(th_top, up['Th'][up['th'].argmax()]),
           arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))
B.annotate('', xy=(th_bot, -0.02), xytext=(th_bot, lo['Th'][lo['th'].argmin()]),
           arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))
B.set_xlabel('θ (heading)'); B.set_ylabel('Θ_neur = arg(γ)')
B.set_title('(B) γ-branches: jumps at the folds (hysteresis)')
B.legend(fontsize=8)

# (C) theta(t): idealized + Euler walker
C = ax[1, 0]
C.plot(ideal_t, np.degrees(ideal_th), color='k', lw=1.5,
       label=f'idealized relaxation osc (period≈{period:.2f})')
C.plot(wt, np.degrees(wth), color='C2', lw=1, alpha=0.8,
       label='warm-started Euler slaved walker')
C.axhline(np.degrees(th_top), color='gray', ls=':', lw=0.8)
C.axhline(np.degrees(th_bot), color='gray', ls=':', lw=0.8)
C.set_xlabel('t'); C.set_ylabel('θ (deg)')
C.set_xlim(0, 8)
C.set_title('(C) Reduced θ(t): small fast relaxation oscillation')
C.legend(fontsize=8)

# (D) gamma-plane
D = ax[1, 1]
D.plot(up['g'].real, up['g'].imag, '-', color='C0', lw=2, label='stable Θ>0')
D.plot(lo['g'].real, lo['g'].imag, '-', color='C3', lw=2, label='stable Θ<0')
D.plot(un['g'].real, un['g'].imag, '--', color='gray', lw=1.5, label='unstable (SC)')
D.plot(wg.real, wg.imag, '.', color='C2', ms=2, alpha=0.4, label='walker γ')
# jump arrows at folds
gt = up['g'][up['th'].argmax()]; gb = lo['g'][lo['th'].argmin()]
D.annotate('', xy=(lo['g'][np.argmin(np.abs(lo['th']-th_top))].real,
                   lo['g'][np.argmin(np.abs(lo['th']-th_top))].imag),
           xytext=(gt.real, gt.imag), arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2))
D.set_xlabel('Re γ'); D.set_ylabel('Im γ')
D.set_title('(D) γ-plane: two stable branches + jump')
D.legend(fontsize=8); D.set_aspect('equal', adjustable='datalim')

fig.suptitle('REDUCED (slaved) dynamics at (2.1, 2.45): γ-bistability relaxation oscillation',
             fontsize=13)
fig.tight_layout()
out = os.path.join(OUTDIR, 'reduced_dynamics_anatomy.png')
fig.savefig(out, dpi=110)
print("saved", out)
