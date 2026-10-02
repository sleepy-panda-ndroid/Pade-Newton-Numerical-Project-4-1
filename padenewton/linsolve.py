"""Linear-solve strategies used inside one Newton-type step.

A strategy has factor(J, fx) which returns an object with two methods:

    solve(B)         used for the Newton direction J^{-1} f(x)
    solve_lambda(B)  used for J^{-1} J(y), which builds the matrix Lambda

For LU, TSVD and LM both methods are the same. LMN (Levenberg-Marquardt on
the Newton direction only) uses the damped solve for the first and a plain LU
solve for the second.

LU    plain LU with partial pivoting
TSVD  truncated SVD (Moore-Penrose): singular values below rtol * sigma_max
      are dropped
LM    (J^T J + lam I)^{-1} J^T B with lam = c * ||f(x)||_2^2, so the damping
      goes to zero at the root (Yamashita and Fukushima, 2001)
LMN   as LM for solve(), exact LU for solve_lambda()
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


class _LUFact:
    def __init__(self, J):
        self.lu = sla.lu_factor(J)

    def solve(self, B):
        return sla.lu_solve(self.lu, B)

    solve_lambda = solve


class _TSVDFact:
    def __init__(self, J, rtol):
        U, s, Vt = np.linalg.svd(J)
        smax = s[0] if s.size else 0.0
        keep = s > rtol * smax if smax > 0 else np.zeros_like(s, dtype=bool)
        self.rank = int(keep.sum())
        self.U, self.Vt = U[:, keep], Vt[keep, :]
        self.sinv = 1.0 / s[keep]

    def solve(self, B):
        Y = self.U.T @ B
        Y = self.sinv * Y if Y.ndim == 1 else self.sinv[:, None] * Y
        return self.Vt.T @ Y

    solve_lambda = solve


class _LMFact:
    def __init__(self, J, lam):
        self.Jt = J.T
        self.lu = sla.lu_factor(J.T @ J + lam * np.eye(J.shape[1]))

    def solve(self, B):
        return sla.lu_solve(self.lu, self.Jt @ B)

    solve_lambda = solve


class _LMNFact:
    def __init__(self, J, lam):
        self.damped = _LMFact(J, lam)
        self.exact = sla.lu_factor(J)

    def solve(self, B):
        return self.damped.solve(B)

    def solve_lambda(self, B):
        return sla.lu_solve(self.exact, B)


class LU:
    name = "LU"

    def factor(self, J, fx=None):
        return _LUFact(J)


class TSVD:
    name = "TSVD"

    def __init__(self, rtol: float = 1e-8):
        self.rtol = rtol

    def factor(self, J, fx=None):
        return _TSVDFact(J, self.rtol)


class _Damped:
    def __init__(self, c: float = 1.0, lam_min: float = 1e-30):
        self.c, self.lam_min = c, lam_min

    def _lam(self, fx):
        return max(self.c * float(np.dot(fx, fx)), self.lam_min)


class LM(_Damped):
    name = "LM"

    def factor(self, J, fx=None):
        return _LMFact(J, self._lam(fx))


class LMN(_Damped):
    name = "LM-N"

    def factor(self, J, fx=None):
        return _LMNFact(J, self._lam(fx))


def make_solver(kind: str, **kw):
    table = {"lu": LU, "tsvd": TSVD, "lm": LM, "lm-n": LMN}
    try:
        return table[kind.lower()](**kw)
    except KeyError:
        raise ValueError(f"unknown solver '{kind}'") from None
