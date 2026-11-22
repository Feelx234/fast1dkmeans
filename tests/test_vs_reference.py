"""Differential test: cluster labels must equal the original numba implementation verbatim."""
import os
import subprocess
import sys

import numpy as np
import pytest

from fast1dkmeans import cluster

OLD_PY = "/tmp/claude-0/venv_old/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = [
    "binary-search-interpolation",
    "binary-search-normal",
    "dynamic-programming-kn",
    "dynamic-programming-space",
]

pytestmark = pytest.mark.skipif(not os.path.exists(OLD_PY), reason="old venv missing")


def make_cases():
    cases = []
    for seed in range(60):
        rng = np.random.RandomState(seed)
        n = int(rng.randint(1, 301)) if seed >= 8 else seed + 1
        kind = seed % 8
        if kind == 0:
            x = rng.rand(n)
        elif kind == 1:
            x = rng.randint(0, 6, n).astype(float)  # many ties
        elif kind == 2:
            x = np.full(n, 3.0)
        elif kind == 3:
            centers = rng.rand(5) * 100
            x = centers[rng.randint(0, 5, n)] + rng.randn(n) * 0.5
        elif kind == 4:
            x = 1e6 + rng.rand(n)
        elif kind == 5:
            x = np.round(rng.rand(n) * 10) / 2  # duplicates on a grid
        elif kind == 6:
            x = rng.randn(n) * 1e3
        else:
            x = rng.randint(-50, 50, n).astype(float)
        cases.append(x)
    rng = np.random.RandomState(999)
    cases.append(rng.rand(5000))
    cases.append(rng.randint(0, 1000, 5000).astype(float))
    return cases


def make_jobs(cases):
    jobs = []
    rng = np.random.RandomState(1)
    for ci, x in enumerate(cases):
        n = len(x)
        if n <= 12:
            ks = list(range(1, n + 1))
        elif n >= 5000:
            ks = [2, 7, 50, 300]
        else:
            ks = sorted(set([1, 2, n - 1, n] + list(rng.randint(1, n + 1, 6))))
            ks = [k for k in ks if 1 <= k <= n]
        for k in ks:
            for mi in range(len(METHODS)):
                jobs.append((ci, k, mi))
    return jobs


def run_old(d, name, jobs, cases, pure=False):
    """Run the original package on `jobs` in a subprocess; returns {job index: labels}.
    pure=True runs it with NUMBA_DISABLE_JIT=1, which turns the out-of-bounds reads of
    the original (see module docstring) into IndexErrors instead of undefined behavior."""
    xs = np.empty(len(cases), dtype=object)
    for i, c in enumerate(cases):
        xs[i] = c
    np.savez(d / f"{name}_cases.npz", xs=xs, meta=np.array(jobs, dtype=np.int64).reshape(-1, 3),
             methods=np.array(METHODS))
    env = dict(os.environ)
    if pure:
        env["NUMBA_DISABLE_JIT"] = "1"
    subprocess.run(
        [OLD_PY, "-I", os.path.join(HERE, "_old_runner.py"), str(d / f"{name}_cases.npz"), str(d / f"{name}_out.npz")],
        check=True, cwd=str(d), env=env,
    )
    out = np.load(d / f"{name}_out.npz")
    return {i: out[str(i)] for i in range(len(jobs))}


def test_labels_identical(tmp_path):
    """New labels == original labels, except where the original has undefined behavior.

    The original `_smawk_iter` reads past its column buffer (and returns garbage / may
    segfault, non-deterministically) whenever the reduce step leaves a single column but
    more than 2 rows remain at that recursion depth. Such inputs are detected by re-running
    the original un-jitted, where the bad read raises IndexError; for those only basic
    validity of the new labels is checked.
    """
    cases = make_cases()
    jobs = make_jobs(cases)
    old = run_old(tmp_path, "jit", jobs, cases)

    def new_labels(ci, k, mi):
        try:
            return np.asarray(cluster(cases[ci], k, METHODS[mi]), dtype=np.int64)
        except BaseException:  # noqa (incl. Rust panics)
            return np.array([-1])

    suspect = []
    for idx, (ci, k, mi) in enumerate(jobs):
        new = new_labels(ci, k, mi)
        if old[idx].shape != new.shape or not np.array_equal(old[idx], new):
            suspect.append(idx)
    bad, undefined = [], 0
    if suspect:
        pure = run_old(tmp_path, "pure", [jobs[i] for i in suspect], cases, pure=True)
        for j, idx in enumerate(suspect):
            ci, k, mi = jobs[idx]
            ref = pure[j]
            new = new_labels(ci, k, mi)
            if ref[0] == -1 and len(ref) == 1 and len(cases[ci]) > 1:
                undefined += 1
                # original undefined: new must still be a valid, ordered, k-cluster labeling
                order = np.argsort(cases[ci], kind="stable")
                nl = new[order]
                if not (len(new) == len(cases[ci]) and (np.diff(nl) >= 0).all() and nl.min() == 0):
                    bad.append(("new invalid", ci, len(cases[ci]), k, METHODS[mi]))
            elif not np.array_equal(ref, new):
                bad.append((ci, len(cases[ci]), k, METHODS[mi]))
    print(f"{len(jobs)} jobs, {len(suspect)} differed from jit run, {undefined} of them original-UB")
    assert not bad, f"{len(bad)} label mismatches where original is well defined, first: {bad[:10]}"
