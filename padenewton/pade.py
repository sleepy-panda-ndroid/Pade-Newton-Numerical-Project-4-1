"""Pade approximants of  phi(z) = 2 / (1 + sqrt(1 - 2 z))  at z = 0.

phi(z) = C(z/2), where C is the Catalan generating function, so the Taylor
coefficients are  c_n = Catalan(n) / 2^n = 1, 1/2, 1/2, 5/8, 7/8, ...

The (p, q) Pade approximant is  A(z)/B(z)  with  deg A <= p,  deg B <= q,
B(0) = 1, and  A/B - phi = O(z^(p+q+1)).  We compute it in exact rational
arithmetic (fractions.Fraction) so the coefficients can be compared exactly
with Table 2 of Herceg et al. (2025).

Coefficient arrays are returned in ASCENDING order of powers.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import comb

import numpy as np


def taylor_coeffs(N: int) -> list[Fraction]:
    """First N+1 Taylor coefficients of 2/(1+sqrt(1-2z))."""
    return [Fraction(comb(2 * n, n), (n + 1) * 2**n) for n in range(N + 1)]


def _solve_fraction(M: list[list[Fraction]], r: list[Fraction]) -> list[Fraction]:
    """Gauss-Jordan elimination in exact arithmetic."""
    n = len(M)
    A = [row[:] + [r[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = next((i for i in range(col, n) if A[i][col] != 0), None)
        if piv is None:
            raise ZeroDivisionError("singular Pade system (degenerate table entry)")
        A[col], A[piv] = A[piv], A[col]
        for i in range(n):
            if i != col and A[i][col] != 0:
                f = A[i][col] / A[col][col]
                A[i] = [a - f * b for a, b in zip(A[i], A[col])]
    return [A[i][n] / A[i][i] for i in range(n)]


@lru_cache(maxsize=None)
def pade_fractions(p: int, q: int) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Exact (numerator, denominator) coefficients, ascending powers, b_0 = 1."""
    if p < 0 or q < 0:
        raise ValueError("p and q must be non-negative")
    c = taylor_coeffs(p + q)
    if q == 0:
        b = [Fraction(1)]
    else:
        M = [[c[k - j] if k - j >= 0 else Fraction(0) for j in range(1, q + 1)]
             for k in range(p + 1, p + q + 1)]
        rhs = [-c[k] for k in range(p + 1, p + q + 1)]
        b = [Fraction(1)] + _solve_fraction(M, rhs)
    a = [sum(b[j] * c[k - j] for j in range(0, min(k, q) + 1)) for k in range(p + 1)]
    return tuple(a), tuple(b)


@lru_cache(maxsize=None)
def pade_coeffs(p: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Floating-point (a, b) coefficient arrays, ascending powers."""
    a, b = pade_fractions(p, q)
    return np.array([float(x) for x in a]), np.array([float(x) for x in b])


def pade_eval(z, p: int, q: int):
    """Evaluate the scalar Pade approximant A(z)/B(z)."""
    a, b = pade_coeffs(p, q)
    return np.polyval(a[::-1], z) / np.polyval(b[::-1], z)


def theoretical_order(p: int, q: int) -> int:
    """Convergence order proved in Theorem 1: min{p+q+2, 4}."""
    return min(p + q + 2, 4)
