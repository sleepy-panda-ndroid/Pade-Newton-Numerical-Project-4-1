"""Shared setup for the experiment scripts.

Importing this module pins BLAS to one thread (so timings are comparable) and
puts the repository root on sys.path. The scripts print nothing except a
one-line start/finish message; all results are written to results/*.csv.
"""
import os
import pathlib
import platform
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
QUICK = "--quick" in sys.argv

_t0 = 0.0


def begin(title):
    global _t0
    _t0 = time.time()
    if not os.environ.get("PADE_RUN_ALL"):
        print(f"{title} ...", flush=True)


def end():
    if not os.environ.get("PADE_RUN_ALL"):
        print(f"  done ({time.time() - _t0:.0f} s)", flush=True)


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()
