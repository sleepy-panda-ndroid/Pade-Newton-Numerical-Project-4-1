# Padé-accelerated Newton methods: efficiency and robustness

CSE402 (Numerical Analysis, Simulation and Modeling Sessional), Section C1.

Section C, Group 10. Submission PDF: `report/C_10.pdf`.

Base paper: Đ. Herceg, D. Herceg, P. Odry, V. Tadić, *A New Family of Methods for Solving
Systems of Nonlinear Equations*, Mathematics 13(24), 3945 (2025).
The paper replaces the square-root term of Cauchy's method by a Padé approximant of order (p,q),
which gives a family of iterations of order min{p+q+2, 4}. (0,0) is Newton and (1,1) is Jarratt.
This project reimplements the family in NumPy and asks two questions the paper does not:

1. Does the higher order pay off in wall-clock time in double precision?
2. What happens when the Jacobian is (nearly) singular, and do truncated SVD or
   Levenberg-Marquardt solves help?

The written report is in `report/` (ACM sigconf format, `report/report.pdf`).

## Setup and run

Python 3.10 or newer.

```
pip install -r requirements.txt
python -m pytest -q tests        # unit tests
python run_all.py --quick        # smoke test, 1-2 minutes
python run_all.py                # full run, roughly 15-25 minutes
```

The terminal only shows which step is running. Every result is written to `results/*.csv`.
Do not run other heavy programs during the full run: the timing experiment is sensitive to it.
BLAS is pinned to one thread inside the scripts.

Each script can also be run alone from `experiments/` (for example `python exp2_robustness.py`).

## Layout

| Path | Content |
|---|---|
| `padenewton/pade.py` | exact Padé coefficients of 2/(1+sqrt(1-2z)) (rational arithmetic) |
| `padenewton/linsolve.py` | LU, truncated SVD, Levenberg-Marquardt (LM) and LM-N solve strategies |
| `padenewton/methods.py` | the (p,q) iteration (generic and cheap forms), counters, cost model |
| `padenewton/systems.py` | test systems (paper Examples 1, 4, 7; H-equation; trig; near-singular; Powell) |
| `padenewton/mpsystems.py` | mpmath versions of Example 1 for high-precision checks |
| `experiments/exp0_validate.py` | iteration counts vs. Table 5 of the paper |
| `experiments/exp1_efficiency.py` | iterations, operation counts, time vs. (p,q) and n |
| `experiments/exp2_robustness.py` | LU / TSVD / LM / LM-N on ill-conditioned problems, TSVD threshold sweep |
| `experiments/exp3_coc.py` | order of convergence in 1500-digit arithmetic |
| `experiments/make_figures.py` | figures and LaTeX tables from the CSV files |
| `tests/` | unit tests |
| `results/` | all numerical output (CSV) |
| `figures/` | plots used in the report |
| `report/` | LaTeX source (`report.tex`, `references.bib`) and the compiled PDF |

## Result files

| File | One row per |
|---|---|
| `validation.csv` | test case of the paper's Table 5 |
| `efficiency.csv` | (system, n, p, q, implementation form) with iterations, counters, median time |
| `robustness_raw.csv` | (system, random start, p, q, solver) |
| `robustness_summary.csv` | the raw rows aggregated over the starts |
| `tsvd_threshold.csv` | (eps, method, TSVD threshold) |
| `coc.csv` | (system, method, iteration) with log10 error and computed order |
| `run_info.csv` | CPU, OS and library versions of the machine that produced the timings |

## Two implementation forms

The direct reading of the paper builds the matrix Λ = 1.5 (I - J(x)⁻¹J(y)), which needs a solve with
n right-hand sides (order n³ work). The code has this as the *generic* form. It also has a *cheap* form,
valid with the LU solver, that gives the same iterates:

* q = 0: Λ is applied to vectors only, Λt = 1.5 (t - J(x)⁻¹(J(y)t)). No matrix is formed.
* (1,1): Jarratt's original step, (6J(y) - 2J(x))⁻¹(3J(y) + J(x))J(x)⁻¹f(x).

`tests/test_core.py` checks that both forms give the same iterates.

## Changes relative to the project proposal

* The proposed near-singular system is linear, so J(y) = J(x), Λ = 0 and every (p,q) method equals Newton.
  A cubic term is added: f2 = (x-1) + (1+ε)(y-1) + (y-1)³. Its Jacobian still has det = ε at the root.
* The chemical-equilibrium system was dropped (it is not among the paper's test problems). Bratu, the
  H-equation, the trigonometric system and the paper's own scalable systems are used instead.
  Singular-Jacobian tests were added: Powell's function, cos(x)-1 and the H-equation at c = 1.
* Paper example (7b) is not used; its formula could not be read reliably from the PDF.
