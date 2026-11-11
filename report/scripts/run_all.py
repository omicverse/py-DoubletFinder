"""Benchmark driver: every task x seed for DoubletFinder (R) and pydoubletfinder, each run pinned to its own
CPU core (taskset), so both are single-threaded and timed under the same load.

Usage: python run_all.py <tasks_dir> <out_dir> <rscript> <python> [n_parallel] [seeds] [engines]
"""
from __future__ import annotations

import os
import pathlib
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
tasks_dir, out_dir, rscript, python = map(str, sys.argv[1:5])
n_par = int(sys.argv[5]) if len(sys.argv) > 5 else 8
seeds = [int(s) for s in (sys.argv[6] if len(sys.argv) > 6 else "1,2").split(",")]
engines = (sys.argv[7] if len(sys.argv) > 7 else "DoubletFinder,pydoubletfinder").split(",")
cores = queue.Queue()
for c in sorted(os.sched_getaffinity(0))[:n_par]:
    cores.put(c)
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OMP_WAIT_POLICY="PASSIVE", PYTHONWARNINGS="ignore")

jobs = []
tasks = sorted(p for p in pathlib.Path(tasks_dir).iterdir() if (p / "counts.mtx.gz").exists())
for t in sorted(tasks, key=lambda p: -p.joinpath("counts.mtx.gz").stat().st_size):   # largest first
    for s in seeds:
        if "DoubletFinder" in engines:
            jobs.append(("DoubletFinder", [rscript, str(HERE / "02_run_doubletfinder.R")], t, s))
        if "pydoubletfinder" in engines:
            jobs.append(("pydoubletfinder", [python, str(HERE / "03_run_pydoubletfinder.py")], t, s))


def run(job):
    engine, cmd, t, s = job
    od = pathlib.Path(out_dir) / engine / t.name / f"seed{s}"
    if (od / "timing.json").exists():
        return
    core = cores.get()
    try:
        r = subprocess.run(["taskset", "-c", str(core), *cmd, str(t), str(od), str(s)], env=env,
                           capture_output=True, text=True)
        print(engine, r.stdout.strip() or r.stderr.strip()[-400:], flush=True)
    finally:
        cores.put(core)


with ThreadPoolExecutor(n_par) as ex:
    list(ex.map(run, jobs))
