"""LU, truncated SVD and Levenberg-Marquardt solves on ill-conditioned problems.

Outputs
  results/robustness_raw.csv        one row per (system, start, method, solver)
  results/robustness_summary.csv    aggregated over the random starts
  results/tsvd_threshold.csv        TSVD threshold sweep on the near-singular family

'accurate' means the forward error to the known root is below 1e-6. For the
H-equation at c=1 there is no reference root, so accurate means the stopping
rule was met and the residual is below 1e-10.
"""
from common import RESULTS, QUICK, begin, end
import numpy as np
import pandas as pd
from padenewton import iterate, systems, make_solver

begin("robustness experiments")

N_TRIALS = 20 if QUICK else 100
EPS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
METHODS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]
SOLVERS = {"LU": lambda: make_solver("lu"),
           "TSVD": lambda: make_solver("tsvd", rtol=1e-8),
           "LM": lambda: make_solver("lm", c=1.0),
           "LM-N": lambda: make_solver("lm-n", c=1.0)}
ACC_TOL = 1e-6
rng = np.random.default_rng(12345)


def cases():
    for eps in EPS:
        for x0 in rng.uniform(-2, 4, size=(N_TRIALS, 2)):
            yield "NearSing", eps, systems.near_singular(eps, x0)
    for _ in range(N_TRIALS):
        yield "Powell", np.nan, systems.powell_singular(np.array([3, -1, 0, 1.]) + rng.uniform(-.5, .5, 4))
    for _ in range(N_TRIALS):
        s = systems.cos_diag(20); s.x0 = rng.uniform(.3, 1.2, 20)
        yield "CosDiag", np.nan, s
    for _ in range(N_TRIALS):
        s = systems.h_equation(50, 1.0); s.x0 = 1 + rng.uniform(0, .2, 50)
        yield "H-eq c=1", np.nan, s


def fwd_error(r, s):
    if s.x_star is None or not np.all(np.isfinite(r.x)):
        return np.nan
    return float(np.max(np.abs(r.x - s.x_star)))


rows = []
for fam, eps, s in cases():
    for p, q in METHODS:
        for sname, make in SOLVERS.items():
            r = iterate(s.F, s.J, s.x0, p, q, solver=make(), maxit=200)
            err = fwd_error(r, s)
            acc = (err < ACC_TOL) if not np.isnan(err) else (r.converged and r.residual < 1e-10)
            rows.append(dict(family=fam, eps=eps, p=p, q=q, solver=sname, converged=r.converged,
                             accurate=bool(acc), status=r.status, iters=r.iters, time=r.time,
                             residual=r.residual, fwd_error=err))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "robustness_raw.csv", index=False)
summ = (df.groupby(["family", "eps", "p", "q", "solver"], dropna=False)
          .agg(runs=("accurate", "size"), converged_rate=("converged", "mean"),
               accurate_rate=("accurate", "mean"), med_iters=("iters", "median"),
               med_time=("time", "median"), med_fwd_error=("fwd_error", "median"),
               med_residual=("residual", "median")).reset_index())
summ.to_csv(RESULTS / "robustness_summary.csv", index=False)

# TSVD threshold sweep (near-singular family, Newton and Jarratt)
rows = []
sweep_rng = np.random.default_rng(2024)
for eps in [1e-4, 1e-6, 1e-8, 1e-10]:
    starts = sweep_rng.uniform(-2, 4, size=(N_TRIALS, 2))
    for p, q in [(0, 0), (1, 1)]:
        for rtol in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]:
            ok, conv, errs = 0, 0, []
            for x0 in starts:
                s = systems.near_singular(eps, x0)
                r = iterate(s.F, s.J, s.x0, p, q, solver=make_solver("tsvd", rtol=rtol), maxit=200)
                e = fwd_error(r, s)
                ok += (not np.isnan(e)) and e < ACC_TOL
                conv += r.converged
                errs.append(e)
            rows.append(dict(eps=eps, p=p, q=q, rtol=rtol, runs=len(starts), accurate_rate=ok / len(starts),
                             converged_rate=conv / len(starts), med_fwd_error=float(np.nanmedian(errs))))
pd.DataFrame(rows).to_csv(RESULTS / "tsvd_threshold.csv", index=False)
end()
