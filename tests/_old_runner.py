"""Run with the ORIGINAL numba package: python _old_runner.py cases.npz out.npz
Each job runs in a forked child because the original can segfault on some inputs
(out-of-bounds access in numba code); a crash is recorded as [-2], a Python
exception as [-1]."""
import os
import sys

import numpy as np
from fast1dkmeans.main import cluster

d = np.load(sys.argv[1], allow_pickle=True)
xs, meta = d["xs"], d["meta"]
methods = list(d["methods"])
# warm up (compile / load cache) in the parent
cluster(np.arange(5.0) ** 2, 2, "binary-search-interpolation")
cluster(np.arange(5.0) ** 2, 2, "binary-search-normal")
cluster(np.arange(5.0) ** 2, 2, "dynamic-programming-kn")
cluster(np.arange(5.0) ** 2, 2, "dynamic-programming-space")
out = {}
tmp = sys.argv[2] + ".tmp"
for idx, (ci, k, mi) in enumerate(meta):
    pid = os.fork()
    if pid == 0:
        try:
            res = np.asarray(cluster(xs[ci], int(k), methods[mi]), dtype=np.int64)
        except Exception:  # noqa
            res = np.array([-1])
        with open(tmp, "wb") as f:
            np.save(f, res)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    if status != 0:
        out[str(idx)] = np.array([-2])
    else:
        out[str(idx)] = np.load(tmp)
    if os.path.exists(tmp):
        os.remove(tmp)
np.savez(sys.argv[2], **out)
