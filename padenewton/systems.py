"""Nonlinear test systems. Each builder returns a System(F, J, x0, x_star)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

S2, S3 = np.sqrt(2.0), np.sqrt(3.0)


@dataclass
class System:
    name: str
    n: int
    F: Callable
    J: Callable
    x0: np.ndarray
    x_star: Optional[np.ndarray] = None


# ---------------- small systems from the base paper (Example 1) -------------
def s1a(x0):
    F = lambda x: np.array([np.exp(x[0]**2) - np.exp(S2 * x[0]), x[0] - x[1]])
    J = lambda x: np.array([[2*x[0]*np.exp(x[0]**2) - S2*np.exp(S2*x[0]), 0.0], [1.0, -1.0]])
    return System("1a", 2, F, J, np.array(x0, float))

def s1b(x0):
    F = lambda x: np.array([x[0] + np.exp(x[1]) - np.cos(x[1]), 3*x[0] - x[1] - np.sin(x[1])])
    J = lambda x: np.array([[1.0, np.exp(x[1]) + np.sin(x[1])], [3.0, -1.0 - np.cos(x[1])]])
    return System("1b", 2, F, J, np.array(x0, float))

def s1c(x0):
    F = lambda x: np.array([x[0]**2 - 2*x[0] - x[1] + 0.5, x[0]**2 + 4*x[1]**2 - 4])
    J = lambda x: np.array([[2*x[0] - 2, -1.0], [2*x[0], 8*x[1]]])
    return System("1c", 2, F, J, np.array(x0, float))

def s1d(x0):
    F = lambda x: np.array([x[0]**2 + x[1]**2 - 1, x[0]**2 - x[1]**2 + 0.5])
    J = lambda x: np.array([[2*x[0], 2*x[1]], [2*x[0], -2*x[1]]])
    return System("1d", 2, F, J, np.array(x0, float))

def s1e(x0):
    F = lambda x: np.array([np.sin(x[0]) + x[1]*np.cos(x[0]), x[0] - x[1]])
    J = lambda x: np.array([[np.cos(x[0]) - x[1]*np.sin(x[0]), np.cos(x[0])], [1.0, -1.0]])
    return System("1e", 2, F, J, np.array(x0, float))

def cyclic_product(n, x0val, name="1f"):
    """f_i = x_i x_{i+1} - 1 (i<n), f_n = x_n x_1 - 1   (paper 1f, odd n)."""
    def F(x):
        return x * np.roll(x, -1) - 1.0
    def J(x):
        M = np.zeros((n, n)); i = np.arange(n)
        M[i, i] = np.roll(x, -1); M[i, (i + 1) % n] = x
        return M
    return System(f"{name}(n={n})", n, F, J, np.full(n, float(x0val)), np.sign(x0val) * np.ones(n))

def s1g(x0):
    def F(x):
        a, b, c, d = x
        return np.array([b*c + d*(b + c), a*c + d*(a + c), a*b + d*(a + b), a*b + a*c + b*c - 1])
    def J(x):
        a, b, c, d = x
        return np.array([[0, c + d, b + d, b + c], [c + d, 0, a + d, a + c],
                         [b + d, a + d, 0, a + b], [b + c, a + c, a + b, 0]], float)
    return System("1g", 4, F, J, np.array(x0, float))


# ---------------- scalable systems -----------------------------------------
def cyclic_cubic(n, x0val=3.0):
    """Paper (7c): f_i = x_i^2 x_{i+1} - 1, f_n = x_n^2 x_1 - 1; root = ones."""
    def F(x):
        return x**2 * np.roll(x, -1) - 1.0
    def J(x):
        M = np.zeros((n, n)); i = np.arange(n)
        M[i, i] = 2*x*np.roll(x, -1); M[i, (i + 1) % n] = x**2
        return M
    return System(f"7c(n={n})", n, F, J, np.full(n, float(x0val)), np.ones(n))

def bratu(n=40, C=3.0, x0val=0.2):
    """Paper (4a): 1-D Bratu problem, second-order finite differences."""
    h = 1.0 / (n + 1)
    def F(x):
        r = -2*x; r[:-1] += x[1:]; r[1:] += x[:-1]
        return r + h*h*C*np.exp(x)
    def J(x):
        M = np.diag(-2 + h*h*C*np.exp(x)); i = np.arange(n - 1)
        M[i, i + 1] = 1.0; M[i + 1, i] = 1.0
        return M
    return System(f"Bratu(n={n})", n, F, J, np.full(n, x0val))

def h_equation(n=100, c=0.99):
    """Chandrasekhar H-equation (radiative transfer). J is singular at the root for c = 1."""
    mu = (np.arange(1, n + 1) - 0.5) / n
    K = mu[:, None] / (mu[:, None] + mu[None, :]); w = c / (2*n)
    D = lambda H: 1.0 - w * (K @ H)
    F = lambda H: H - 1.0 / D(H)
    J = lambda H: np.eye(n) - (w / D(H)**2)[:, None] * K
    return System(f"H-eq(n={n},c={c})", n, F, J, np.ones(n))

def trig_system(n, x0val=None):
    """More-Garbow-Hillstrom trigonometric system (dense Jacobian)."""
    idx = np.arange(1, n + 1)
    x0val = 1.0 / n if x0val is None else x0val
    F = lambda x: n - np.sum(np.cos(x)) + idx*(1 - np.cos(x)) - np.sin(x)
    def J(x):
        M = np.tile(np.sin(x), (n, 1))
        M[np.arange(n), np.arange(n)] += idx*np.sin(x) - np.cos(x)
        return M
    return System(f"Trig(n={n})", n, F, J, np.full(n, x0val))


# ---------------- ill-conditioned / singular-Jacobian systems ---------------
def near_singular(eps, x0=(3.0, -1.0)):
    """f1 = x + y - 2,  f2 = (x-1) + (1+eps)(y-1) + (y-1)^3.  Root (1,1).

    det J = eps + 3 (y-1)^2, so det J(root) = eps. The purely linear system from
    the proposal has J(y) = J(x), hence Lambda = 0 and every (p,q) method is
    Newton; the cubic term avoids that."""
    F = lambda x: np.array([x[0] + x[1] - 2, (x[0]-1) + (1+eps)*(x[1]-1) + (x[1]-1)**3])
    J = lambda x: np.array([[1.0, 1.0], [1.0, 1 + eps + 3*(x[1]-1)**2]])
    return System(f"NearSing(eps={eps:g})", 2, F, J, np.array(x0, float), np.ones(2))

def powell_singular(x0=(3.0, -1.0, 0.0, 1.0)):
    s5, s10 = np.sqrt(5.0), np.sqrt(10.0)
    F = lambda x: np.array([x[0] + 10*x[1], s5*(x[2] - x[3]), (x[1] - 2*x[2])**2, s10*(x[0] - x[3])**2])
    def J(x):
        u, w = 2*(x[1] - 2*x[2]), 2*s10*(x[0] - x[3])
        return np.array([[1, 10, 0, 0], [0, 0, s5, -s5], [0, u, -2*u, 0], [w, 0, 0, -w]], float)
    return System("Powell", 4, F, J, np.array(x0, float), np.zeros(4))

def cos_diag(n=20, x0val=0.78):
    """Paper (7a): f_i = cos(x_i) - 1. J(root) = 0."""
    F = lambda x: np.cos(x) - 1.0
    J = lambda x: np.diag(-np.sin(x))
    return System(f"CosDiag(n={n})", n, F, J, np.full(n, x0val), np.zeros(n))


# ---------------- Table-5 validation cases ---------------------------------
# (label, System, expected iterations: Newton, PM(3)=(0,1), best PM(4))
def paper_cases():
    return [
        ("1a-1", s1a([2.3, 2.3]), 10, 7, 4), ("1a-2", s1a([1.8, 1.8]), 7, 5, 4),
        ("1a-3", s1a([0.8, 0.8]), 5, 4, 4),
        ("1b-1", s1b([1.5, 2.0]), 7, 5, 4), ("1b-2", s1b([0.3, 0.5]), 5, 4, 3),
        ("1c-1", s1c([3.0, 2.0]), 7, 5, 4), ("1c-2", s1c([1.6, 0.0]), 5, 4, 3),
        ("1d-1", s1d([0.7, 1.2]), 5, 4, 3), ("1d-2", s1d([-1.0, -2.0]), 6, 5, 3),
        ("1e-1", s1e([1.2, -1.5]), 6, 4, 4), ("1e-2", s1e([-0.6, 0.6]), 5, 3, 3),
        ("1f-1", cyclic_product(99, 2.0), 6, 4, 3), ("1f-2", cyclic_product(99, -4.0), 7, 5, 3),
        ("1g-1", s1g([-1, -1, -1, -1]), 6, 4, 3), ("1g-2", s1g([2, 2, 2, 0]), 7, 5, 3),
    ]
