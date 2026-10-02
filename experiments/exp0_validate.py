"""Compare iteration counts with Table 5 of the base paper.

Columns written to results/validation.csv:
  newton, pm3, pm4_best   our double-precision iteration counts
  *_paper                 the counts printed in the paper
  newton_mp50             plain Newton in 50-digit arithmetic (2-D/4-D cases),
                          to check whether float64 explains a mismatch
"""
from common import RESULTS, begin, end
import mpmath as mp
import pandas as pd
from padenewton import iterate, systems
from padenewton.mpsystems import newton_iterations

begin("validation against the base paper")
mp.mp.dps = 50

rows = []
for lab, s, n_paper, pm3_paper, pm4_paper in systems.paper_cases():
    it = lambda p, q: iterate(s.F, s.J, s.x0, p, q).iters
    best = min((it(p, q), (p, q)) for p in range(5) for q in range(5) if p + q > 1)
    key = lab.split("-")[0]
    n_mp = newton_iterations(key, s.x0) if key in ("1a", "1b", "1c", "1d", "1e", "1g") else None
    rows.append(dict(case=lab, newton=it(0, 0), newton_paper=n_paper, newton_mp50=n_mp,
                     pm3=it(0, 1), pm3_paper=pm3_paper,
                     pm4_best=best[0], pm4_best_pq=str(best[1]), pm4_paper=pm4_paper))
df = pd.DataFrame(rows)
df["match"] = (df.newton == df.newton_paper) & (df.pm3 == df.pm3_paper) & (df.pm4_best == df.pm4_paper)
df.to_csv(RESULTS / "validation.csv", index=False)
end()
