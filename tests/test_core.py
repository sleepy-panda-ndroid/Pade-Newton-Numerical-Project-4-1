import pathlib
import sys
from fractions import Fraction as Fr

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from padenewton import iterate, make_solver, pade_eval, pade_fractions, systems


def test_pade_coefficients_match_table2():
    a, b = pade_fractions(1, 1)               # (2 - z) / (2 - 2z)
    assert [x / b[0] for x in a] == [Fr(1), Fr(-1, 2)] and b[1] / b[0] == Fr(-1)
    a, b = pade_fractions(0, 1)               # 2 / (2 - z)
    assert a == (Fr(1),) and b == (Fr(1), Fr(-1, 2))
    a, b = pade_fractions(2, 0)
    assert a == (Fr(1), Fr(1, 2), Fr(1, 2))


def test_pade_agrees_with_function_to_order_p_plus_q():
    z = 1e-3
    exact = 2 / (1 + np.sqrt(1 - 2 * z))
    for p, q in [(1, 1), (2, 1), (2, 2), (3, 1)]:
        assert abs(pade_eval(z, p, q) - exact) < 10 * z ** (p + q + 1)


def test_all_small_members_converge():
    s = systems.s1g([-1, -1, -1, -1])
    for pq in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]:
        r = iterate(s.F, s.J, s.x0, *pq)
        assert r.converged and r.residual < 1e-10


def test_cheap_form_gives_same_iterates_as_generic():
    for s in [systems.s1g([-1, -1, -1, -1]), systems.cyclic_cubic(21, 3.0), systems.bratu(30)]:
        for p, q in [(1, 0), (2, 0), (1, 1)]:
            g = iterate(s.F, s.J, s.x0, p, q, form="generic")
            c = iterate(s.F, s.J, s.x0, p, q, form="cheap")
            assert g.iters == c.iters
            assert np.allclose(g.x, c.x, atol=1e-10)


def test_solvers_agree_on_well_conditioned_problem():
    s = systems.s1b([0.3, 0.5])
    for kind in ["lu", "tsvd", "lm", "lm-n"]:
        r = iterate(s.F, s.J, s.x0, 1, 1, solver=make_solver(kind))
        assert r.converged and np.allclose(r.x, 0, atol=1e-8)


def test_lm_n_uses_exact_solve_for_lambda():
    s = systems.near_singular(1e-2, [1.3, 0.7])
    x = np.array([1.3, 0.7])
    J, fx = s.J(x), s.F(x)
    fac = make_solver("lm-n").factor(J, fx)
    B = s.J(x - 2 / 3 * np.linalg.solve(J, fx))
    assert np.allclose(fac.solve_lambda(B), np.linalg.solve(J, B))
    assert not np.allclose(fac.solve(fx), np.linalg.solve(J, fx), atol=1e-12)
