"""Pick a default value for `e` in PerceptionModel's `reg_power` neural weight.

The reg_power weight is f(theta; d, e) = 1 / (|theta|^d + e); its normalized
integral F(theta; d, e) (computed by _reg_power_integral, mapping [-pi, pi] to
itself) converges to _power(theta, c=1-d) as e -> 0. We sweep e and report:

  err_pow(d, e)    = max |F(theta; d, e) - _power(theta, 1-d)| over theta in [-pi, pi]
  err_sym(d, e)    = max |F(theta; d, e) + F(-theta; d, e)| (symmetry sanity)

The recommended default is the smallest e for which err_pow at d=0.5 falls below
~1e-3 with no numerical issues. The pinned regression test in
test_perception_spline.py uses the value reported here.
"""

import numpy as np
from decision_model import PerceptionModel as PM


def err_vs_power(d, e, n=2001):
    theta = np.linspace(-np.pi, np.pi, n)
    F = PM._reg_power_integral(theta, d, e)
    P = PM._power(theta, 1.0 - d)
    return float(np.max(np.abs(F - P)))


def err_symmetry(d, e, n=501):
    theta = np.linspace(0.0, np.pi, n)
    F_pos = PM._reg_power_integral(theta, d, e)
    F_neg = PM._reg_power_integral(-theta, d, e)
    return float(np.max(np.abs(F_pos + F_neg)))


def main():
    e_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    d_values = [0.3, 0.5, 0.7]

    print("=" * 64)
    print(" max |F(theta; d, e) - _power(theta, c=1-d)| on [-pi, pi]")
    print("=" * 64)
    header = f"{'e':>10s}" + "".join(f"  d={d:<5g}  " for d in d_values)
    print(header)
    print("-" * len(header))
    for e in e_values:
        row = f"{e:>10.0e}"
        for d in d_values:
            row += f"  {err_vs_power(d, e):.3e}  "
        print(row)

    print()
    print("=" * 64)
    print(" max |F(theta) + F(-theta)| (symmetry residual)")
    print("=" * 64)
    print(header)
    print("-" * len(header))
    for e in e_values:
        row = f"{e:>10.0e}"
        for d in d_values:
            row += f"  {err_symmetry(d, e):.3e}  "
        print(row)

    print()
    print("Recommendation: pick e where err_pow at d=0.5 is around 1e-3 or below.")
    print("Going smaller costs little; going much larger visibly distorts the curve")
    print("relative to _power. e = 1e-3 gives ~7e-3 error, e = 1e-4 gives ~1e-3.")


if __name__ == "__main__":
    main()
