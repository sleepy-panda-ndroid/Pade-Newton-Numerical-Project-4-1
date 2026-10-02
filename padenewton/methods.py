"""The (p,q) Pade-Newton family of Herceg et al. (2025).

    y      = x - (2/3) J(x)^{-1} f(x)
    Lambda = (3/2) (I - J(x)^{-1} J(y))
    x+     = x - B(Lambda)^{-1} A(Lambda) J(x)^{-1} f(x)

with A/B the (p,q) Pade approximant of 2/(1 + sqrt(1 - 2z)). (0,0) is Newton
and (1,1) is Jarratt's method.

Two ways of computing a step are implemented. Both give the same iterates up
to round-off.

  generic  builds Lambda explicitly (one solve with n right-hand sides) and
           forms B(Lambda) when q >= 1. Works for every (p,q) and every
           linear-solve strategy.
  cheap    only for the LU solver. For q = 0 Lambda is never formed: it is
           applied to vectors with  Lambda t = 1.5 (t - J(x)^{-1} (J(y) t)).
           For (1,1) the step is Jarratt's original
           (6 J(y) - 2 J(x))^{-1} (3 J(y) + J(x)) J(x)^{-1} f(x).

Stopping rule (same as the paper):  ||f(x_k)||_inf + ||x_k - x_{k-1}||_inf < tol.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from .linsolve import LU
from .pade import pade_coeffs


@dataclass
class Counters:
    nf: int = 0           # function evaluations
    nJ: int = 0           # Jacobian evaluations
    nfact: int = 0        # matrix factorisations
    nsolve: int = 0       # solves with one right-hand side
    nsolve_mat: int = 0   # solves with n right-hand sides
    nmatvec: int = 0
    nmatmul: int = 0


@dataclass
class Result:
    x: np.ndarray
    converged: bool
    status: str
    iters: int
    residual: float
    step: float
    time: float
    counters: Counters
    xs: list = field(default_factory=list)


def is_cheap(p: int, q: int) -> bool:
    return q == 0 or (p, q) == (1, 1)


def ops_per_iter(n: int, p: int, q: int, form: str = "generic") -> float:
    """Multiplications and divisions per iteration, leading terms only."""
    lu, tri = n**3 / 3, n**2
    if p == 0 and q == 0:
        return lu + tri
    if form == "cheap" and q == 0:
        return lu + tri + 2 * p * n**2
    if form == "cheap" and (p, q) == (1, 1):
        return 2 * lu + 4 * n**2
    ops = lu + tri + n**3 + p * n**2
    if q > 0:
        ops += max(q - 1, 0) * n**3 + lu + tri
    return ops


def _step(J, x, fx, p, q, a, b, solver, form, c: Counters):
    n = x.size
    Jx = J(x); c.nJ += 1
    fac = solver.factor(Jx, fx); c.nfact += 1
    v = fac.solve(fx); c.nsolve += 1
    if p == 0 and q == 0:
        return x - v
    y = x - (2.0 / 3.0) * v
    Jy = J(y); c.nJ += 1

    if form == "cheap" and q == 0:
        t, Av = v, a[0] * v
        for ak in a[1:]:
            t = 1.5 * (t - fac.solve_lambda(Jy @ t))
            c.nmatvec += 1; c.nsolve += 1
            Av = Av + ak * t
        return x - Av

    if form == "cheap" and (p, q) == (1, 1):
        M = 6.0 * Jy - 2.0 * Jx
        rhs = 3.0 * (Jy @ v) + Jx @ v; c.nmatvec += 2
        c.nfact += 1; c.nsolve += 1
        return x - sla.lu_solve(sla.lu_factor(M), rhs)

    Lam = 1.5 * (np.eye(n) - fac.solve_lambda(Jy)); c.nsolve_mat += 1
    t, Av = v, a[0] * v
    for ak in a[1:]:
        t = Lam @ t; c.nmatvec += 1
        Av = Av + ak * t
    if q == 0:
        return x - Av
    B = b[0] * np.eye(n) + b[1] * Lam
    P = Lam
    for bk in b[2:]:
        P = P @ Lam; c.nmatmul += 1
        B = B + bk * P
    w = sla.lu_solve(sla.lu_factor(B), Av); c.nfact += 1; c.nsolve += 1
    return x - w


def iterate(F, J, x0, p=0, q=0, solver=None, tol=1e-12, maxit=100,
            form="generic", keep_history=False, diverge=1e10) -> Result:
    solver = solver or LU()
    if form == "cheap" and not isinstance(solver, LU):
        raise ValueError("the cheap form is only implemented for the LU solver")
    a, b = pade_coeffs(p, q)
    c = Counters()
    x = np.array(x0, dtype=float)
    xs = [x.copy()] if keep_history else []
    status, conv, k, res, step = "maxit", False, 0, np.inf, np.inf
    t0 = time.perf_counter()
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fx = F(x); c.nf += 1
            for k in range(1, maxit + 1):
                try:
                    xn = _step(J, x, fx, p, q, a, b, solver, form, c)
                except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                    status = "breakdown"; break
                fn = F(xn); c.nf += 1
                if not (np.all(np.isfinite(xn)) and np.all(np.isfinite(fn))):
                    status = "nonfinite"; break
                step = float(np.max(np.abs(xn - x))); res = float(np.max(np.abs(fn)))
                x, fx = xn, fn
                if keep_history:
                    xs.append(x.copy())
                if res + step < tol:
                    conv, status = True, "converged"; break
                if np.max(np.abs(x)) > diverge:
                    status = "diverged"; break
        except (FloatingPointError, OverflowError):
            status = "nonfinite"
    return Result(x, conv, status, k, res, step, time.perf_counter() - t0, c, xs)


def newton(F, J, x0, **kw):
    return iterate(F, J, x0, 0, 0, **kw)
