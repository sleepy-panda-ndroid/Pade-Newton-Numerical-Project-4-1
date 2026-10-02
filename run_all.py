"""Run every experiment, then build figures and report tables.

    python run_all.py            full run (about 15-25 minutes)
    python run_all.py --quick    small smoke test (about 1-2 minutes)

The terminal only shows which step is running. Results go to results/*.csv,
figures to figures/, LaTeX tables to report/generated/.
"""
import os
import pathlib
import subprocess
import sys
import time

here = pathlib.Path(__file__).resolve().parent / "experiments"
flags = [a for a in sys.argv[1:] if a == "--quick"]
steps = [("exp0_validate", "validation against the base paper"),
         ("exp1_efficiency", "efficiency benchmark"),
         ("exp2_robustness", "robustness experiments"),
         ("exp3_coc", "order of convergence check"),
         ("make_figures", "figures and report tables")]

env = dict(os.environ, PADE_RUN_ALL="1")
for i, (script, title) in enumerate(steps, 1):
    print(f"[{i}/{len(steps)}] {title} ...", end=" ", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(here / f"{script}.py")] + flags,
                          cwd=here, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print("FAILED")
        print(proc.stderr[-2000:])
        sys.exit(1)
    print(f"done ({time.time() - t0:.0f} s)")
print("finished. results/ figures/ report/generated/ are up to date.")
