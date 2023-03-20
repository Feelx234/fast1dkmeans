[![build](https://github.com/Feelx234/fast1dkmeans/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/Feelx234/fast1dkmeans/actions)

fast1dkmeans
========

A Python library which implements several variations of optimal *k*-means clustering on 1D data, based on the algorithms presented by Gronlund et al. (2017). This package is inspired by the [kmeans1d](https://github.com/dstein64/kmeans1d) package but extends it by implementing additional algorithms, in particular those with reduced memory requirements O(n) instead of O(kn).

There are several different ways to compute the optimal k-means clustering in 1d.
Currently the package implements the following methods:
- `"binary-search-interpolation"` *default* [O(n lg(U) ), O(n) space, "wilber-interpolation"]
- `"dynamic-programming-kn"` [O(kn), O(kn) space]
- `"dynamic-programming-space"` [O(kn), O(n) space, "dp-linear"]
- `"binary-search-normal"` [O(n lg(U) ), O(n) space, section 2.4, "wilber-binary"]



The code is written in Python and relies on the [numba](https://numba.pydata.org/) compiler for speed.

Requirements
------------

*fast1dkmeans* relies on `numpy` and `numba` which currently support python 3.8-3.10.

Installation
------------

[fast1dkmeans](https://pypi.python.org/pypi/fast1dkmeans) is available on PyPI, the Python Package Index.

```sh
$ pip3 install fast1dkmeans
```

Example Usage
-------------

```python
import fast1dkmeans

x = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
k = 4

clusters = fast1dkmeans.cluster(x, k)

print(clusters)   # [1, 1, 1, 0, 3, 3, 3, 2, 2, 2]
```

Important notice: On first usage the the code is compiled once which may take about 30s. On subsequent usages this is no longer necessary and execution is much faster.

Tests
-----

Tests are in [tests/](https://github.com/Feelx234/fast1dkmeans/blob/master/tests).

```sh
# Run tests
$ python3 -m pytest .
```

License
-------

The code in this repository has an BSD 2-Clause "Simplified" License.

See [LICENSE](https://github.com/Feelx234/fast1dkmeans/blob/master/LICENSE).

References
----------

[1] Gronlund, Allan, Kasper Green Larsen, Alexander Mathiasen, Jesper Sindahl Nielsen, Stefan Schneider,
and Mingzhou Song. "Fast Exact K-Means, k-Medians and Bregman Divergence Clustering in 1D."
ArXiv:1701.07204 [Cs], January 25, 2017. http://arxiv.org/abs/1701.07204.
