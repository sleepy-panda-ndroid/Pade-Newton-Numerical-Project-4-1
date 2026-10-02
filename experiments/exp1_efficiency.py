"""Iterations, operation counts and wall-clock time against (p,q) and n.

Main run (n <= 401): every method with the generic implementation; the members
that have a cheaper form ((1,0), (2,0), (1,1)) are run a second time with
form="cheap".
Large-n run (n = 601..1601): Newton and the cheap forms only, because the
generic form takes too long there.

Timing: within each (system, n) the methods are run in turns, REPEATS times, so
that slow drifts in machine load affect every method equally.
Outputs: results/efficiency.csv, results/efficiency_large.csv, results/run_info.csv
"""
from common import RESULTS, QUICK, begin, end, cpu_name
import platform
import numpy as np
import pandas as pd
import scipy
from padenewton import iterate, systems, ops_per_iter, theoretical_order

begin("efficiency benchmark")

METHODS = [(0, 0, "generic"), (1, 0, "generic"), (0, 1, "generic"), (1, 1, "generic"),
           (2, 0, "generic"), (2, 1, "generic"), (1, 2, "generic"),
           (1, 0, "cheap"), (2, 0, "cheap"), (1, 1, "cheap")]
METHODS_LARGE = [(0, 0, "generic"), (1, 0, "cheap"), (2, 0, "cheap"), (1, 1, "cheap")]
SIZES = [21, 51, 101] if QUICK else [21, 51, 101, 201, 301, 401]
SIZES_LARGE = [201, 301] if QUICK else [601, 801, 1201, 1601]
REPEATS = 3 if QUICK else 15
REPEATS_LARGE = 2 if QUICK else 5

FAMILIES = {
    "1f cyclic product": lambda n: systems.cyclic_product(n, 2.0),
    "7c cyclic cubic": lambda n: systems.cyclic_cubic(n, 3.0),
    "Bratu": lambda n: systems.bratu(n),
    "H-equation c=0.99": lambda n: systems.h_equation(n, 0.99),
    "Trig (x0=1/n)": lambda n: systems.trig_system(n, 1.0 / n),
}


def measure(families, sizes, methods, repeats):
    rows = []
    for fam, build in families.items():
        for n in sizes:
            n_eff = n if n % 2 == 1 or not fam.startswith(("1f", "7c")) else n + 1
            s = build(n_eff)
            runs = {}
            for p, q, form in methods:                 # warm-up run, also gives the counts
                runs[(p, q, form)] = iterate(s.F, s.J, s.x0, p, q, maxit=60, form=form)
            times = {m: [] for m in methods}
            for _ in range(repeats):
                for p, q, form in methods:
                    times[(p, q, form)].append(iterate(s.F, s.J, s.x0, p, q, maxit=60, form=form).time)
            for p, q, form in methods:
                r = runs[(p, q, form)]; c = r.counters; ts = times[(p, q, form)]
                rows.append(dict(system=fam, n=n_eff, p=p, q=q, form=form, order=theoretical_order(p, q),
                                 converged=r.converged, status=r.status, iters=r.iters,
                                 nf=c.nf, nJ=c.nJ, nfact=c.nfact, nsolve=c.nsolve,
                                 nsolve_mat=c.nsolve_mat, nmatmul=c.nmatmul, residual=r.residual,
                                 time_median=float(np.median(ts)), time_min=float(np.min(ts)),
                                 time_q1=float(np.percentile(ts, 25)), time_q3=float(np.percentile(ts, 75)),
                                 ops_per_iter=ops_per_iter(n_eff, p, q, form)))
    df = pd.DataFrame(rows)
    df["ops_total"] = df.ops_per_iter * df.iters
    return df


measure(FAMILIES, SIZES, METHODS, REPEATS).to_csv(RESULTS / "efficiency.csv", index=False)
large = {k: v for k, v in FAMILIES.items() if not k.startswith("Trig")}     # Trig: nothing but Newton and (1,1) converge
measure(large, SIZES_LARGE, METHODS_LARGE, REPEATS_LARGE).to_csv(RESULTS / "efficiency_large.csv", index=False)

info = {"cpu": cpu_name(), "os": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "blas_threads": 1,
        "timing_repeats": REPEATS, "timing_repeats_large_n": REPEATS_LARGE, "quick_run": QUICK}
pd.DataFrame(list(info.items()), columns=["key", "value"]).to_csv(RESULTS / "run_info.csv", index=False)
end()
