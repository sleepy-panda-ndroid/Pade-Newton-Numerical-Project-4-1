"""Base-paper Example 1 systems written for mpmath (high-precision checks)."""
import mpmath as mp

M = mp.matrix
s2 = lambda: mp.sqrt(2)


def _1a():
    F = lambda x: M([mp.exp(x[0]**2) - mp.exp(s2()*x[0]), x[0] - x[1]])
    J = lambda x: M([[2*x[0]*mp.exp(x[0]**2) - s2()*mp.exp(s2()*x[0]), 0], [1, -1]])
    return F, J


def _1b():
    F = lambda x: M([x[0] + mp.exp(x[1]) - mp.cos(x[1]), 3*x[0] - x[1] - mp.sin(x[1])])
    J = lambda x: M([[1, mp.exp(x[1]) + mp.sin(x[1])], [3, -1 - mp.cos(x[1])]])
    return F, J


def _1c():
    F = lambda x: M([x[0]**2 - 2*x[0] - x[1] + mp.mpf(1)/2, x[0]**2 + 4*x[1]**2 - 4])
    J = lambda x: M([[2*x[0] - 2, -1], [2*x[0], 8*x[1]]])
    return F, J


def _1d():
    F = lambda x: M([x[0]**2 + x[1]**2 - 1, x[0]**2 - x[1]**2 + mp.mpf(1)/2])
    J = lambda x: M([[2*x[0], 2*x[1]], [2*x[0], -2*x[1]]])
    return F, J


def _1e():
    F = lambda x: M([mp.sin(x[0]) + x[1]*mp.cos(x[0]), x[0] - x[1]])
    J = lambda x: M([[mp.cos(x[0]) - x[1]*mp.sin(x[0]), mp.cos(x[0])], [1, -1]])
    return F, J


def _1g():
    F = lambda x: M([x[1]*x[2] + x[3]*(x[1] + x[2]), x[0]*x[2] + x[3]*(x[0] + x[2]),
                     x[0]*x[1] + x[3]*(x[0] + x[1]), x[0]*x[1] + x[0]*x[2] + x[1]*x[2] - 1])
    J = lambda x: M([[0, x[2]+x[3], x[1]+x[3], x[1]+x[2]], [x[2]+x[3], 0, x[0]+x[3], x[0]+x[2]],
                     [x[1]+x[3], x[0]+x[3], 0, x[0]+x[1]], [x[1]+x[2], x[0]+x[2], x[0]+x[1], 0]])
    return F, J


BUILDERS = {"1a": _1a, "1b": _1b, "1c": _1c, "1d": _1d, "1e": _1e, "1g": _1g}


def newton_iterations(name, x0, tol=mp.mpf(10)**-12, maxit=60):
    """Iterations of plain Newton with the paper's stopping rule, in mpmath arithmetic."""
    F, J = BUILDERS[name]()
    x = M([mp.mpf(v) for v in x0])
    for k in range(1, maxit + 1):
        xn = x - mp.lu_solve(J(x), F(x))
        res = max(abs(v) for v in F(xn))
        step = max(abs(xn[i] - x[i]) for i in range(len(x)))
        x = xn
        if res + step < tol:
            return k
    return maxit
