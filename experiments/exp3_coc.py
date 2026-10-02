"""Computational order of convergence in 1500-digit arithmetic.

COC_k = ln(e_{k+1}/e_k) / ln(e_k/e_{k-1}),  e_k = ||x_k - alpha||_inf.
Output: results/coc.csv (log10 of the error and the COC at every iteration).
"""
from common import RESULTS, begin, end
import mpmath as mp
import pandas as pd
from padenewton.pade import pade_fractions
from padenewton.mpsystems import BUILDERS

begin("order of convergence check")
mp.mp.dps = 1500
mpf = mp.mpf
STARTS = {"1a": [1.8, 1.8], "1b": [0.3, 0.5], "1g": [-1, -1, -1, -1]}
METHODS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (1, 2)]


def nrm(v):
    return max(abs(v[i]) for i in range(len(v)))


def step(F, J, x, a, b):
    n = len(x)
    Ji = mp.inverse(J(x)); v = Ji * F(x)
    if len(a) == 1 and len(b) == 1:
        return x - v
    y = x - mpf(2) / 3 * v
    Lam = mpf(3) / 2 * (mp.eye(n) - Ji * J(y))
    t, Av = v, a[0] * v
    for ak in a[1:]:
        t = Lam * t; Av = Av + ak * t
    B = b[0] * mp.eye(n); P = mp.eye(n)
    for bk in b[1:]:
        P = P * Lam; B = B + bk * P
    return x - mp.lu_solve(B, Av)


rows = []
for name, x0 in STARTS.items():
    F, J = BUILDERS[name]()
    x = mp.matrix([mpf(v) for v in x0])
    for _ in range(90):
        x = x - mp.inverse(J(x)) * F(x)          # reference root
    alpha = x
    for p, q in METHODS:
        af, bf = pade_fractions(p, q)
        a = [mpf(f.numerator) / f.denominator for f in af]
        b = [mpf(f.numerator) / f.denominator for f in bf]
        x = mp.matrix([mpf(v) for v in x0]); errs = [nrm(x - alpha)]
        for _ in range(14):
            x = step(F, J, x, a, b); errs.append(nrm(x - alpha))
            if errs[-1] < mpf(10) ** (-1000):
                break
        for k in range(2, len(errs)):
            if errs[k] > 0 and errs[k - 1] > 0 and errs[k - 2] > 0:
                coc = mp.log(errs[k] / errs[k - 1]) / mp.log(errs[k - 1] / errs[k - 2])
                rows.append(dict(system=name, p=p, q=q, k=k, log10_err=float(mp.log10(errs[k])), coc=float(coc)))
pd.DataFrame(rows).to_csv(RESULTS / "coc.csv", index=False)
end()
