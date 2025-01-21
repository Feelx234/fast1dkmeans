[![build](https://github.com/Feelx234/fast1dkmeans/actions/workflows/ci.yaml/badge.svg)](https://github.com/Feelx234/fast1dkmeans/actions)
[![Coverage Status](https://coveralls.io/repos/github/Feelx234/fast1dkmeans/badge.svg)](https://coveralls.io/github/Feelx234/fast1dkmeans)

fast1dkmeans
========

A Python library which implements several algorithms to optimally solve *k*-means clustering on 1D data.
Unlike in higher dimensions, the optimal solutions can be found efficiently.
The selection of algorithms is based on those presented by Gronlund et al. (2017).

This package is inspired by the [kmeans1d](https://github.com/dstein64/kmeans1d) package but improves it by implementing additional algorithms with memory requirements of $O(n)$ instead of $O(kn)$. This makes it easier and faster to use *fast1dkmeans* for larger values on $n$ and $k$.

Currently this package implements the following algorithms:
- `"binary-search-interpolation"` *default* [O(n lg(U)), O(n) space]
- `"dynamic-programming-kn"` [O(kn), O(kn) space]
- `"dynamic-programming-space"` [O(kn), O(n) space]
- `"binary-search-normal"` [O(n lg(U) ), O(n) space]

All the methods rely on first sorting the values to be clustered which is omitted in the runtime analysis.

The code is written in Python but all the number crunching is done in compiled code. To achieve this, this project relies on the [numba](https://numba.pydata.org/) compiler for speed.

Requirements
------------

*fast1dkmeans* relies on `numpy` and `numba` and is tested on python 3.9-3.11.

Installation
------------

[fast1dkmeans](https://pypi.python.org/pypi/fast1dkmeans) is available on PyPI, the Python Package Index. It can thus be installed by the following:


```sh
$ pip3 install fast1dkmeans
```

Example Usage
-------------

A simple use of this package is shown below, where we want to cluster the list of values in `x` into four (`k = 4`) clusters.
The optimal clustering of `x` into four groups is pretty obvious as there are essentially four groups of values.
One group around 4.1, one group around -50, one group around 200 and the last group around 100. Let us use *fast1dkmeans* to find the optimal clustering.

```python
import fast1dkmeans

x = [4.0, 4.1, 4.2, -50, 201, 200.4, 80, 102, 100, 200.9, 200.2]
k = 4

clusters = fast1dkmeans.cluster(x, k)

print(clusters)
 # [1, 1, 1, 0, 3, 3, 2, 2, 2, 3, 3]
```

The resulting array `clusters` consists of integers indicating the cluster memberships of values in `x`.
The first three values of `clusters` (three ones) indicate that the first three values of `x` (`[4.0, 4.1, 4.2]`) should be its own cluster (these are the only ones in `clusters`).
The fourth value of `clusters` is the only zero and shows that the fourth value of `x` (-50) should be it's own cluster.
The threes (`3`) in `clusters` indicate that the values `[200.2, 200.4, 200.9, 201]` should be one cluster. Lastly the remaining two's (`2`) form the last cluster of the values [80,100,102].


A different method of clustering can be chosen by passing a keyword argument. Below we for example choose the space reduced dynamic program.
```python
clusters = fast1dkmeans.cluster(x, k, method='dynamic-programming-space')
print(clusters)
 # [1, 1, 1, 0, 3, 3, 2, 2, 2, 3, 3]
```

All the algorithms will return one optimal clustering (of the potentially many) but they runtime and space requirements are very different.


*Important notice*: On first usage the the code is compiled once which may take about 30s. On subsequent usages this is no longer necessary and execution is much faster.



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
