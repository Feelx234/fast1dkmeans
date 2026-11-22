import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fast1dkmeans import cluster
from fast1dkmeans.main import undo_argsort

METHODS = (
    "binary-search-interpolation",
    "binary-search-normal",
    "dynamic-programming-kn",
    "dynamic-programming-space",
    "dynamic-programming",
)

ARR = np.array(
    [0.01864729, 0.23297427, 0.38786064, 0.56103022, 0.74712164, 0.80063267, 0.8071052, 0.86354185]
)
SOLUTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 2, 2, 2, 2],
    [0, 1, 1, 2, 3, 3, 3, 3],
    [0, 1, 2, 3, 4, 4, 4, 4],
    [0, 1, 2, 3, 4, 4, 4, 5],
    [0, 1, 2, 3, 4, 5, 5, 6],
    [0, 1, 2, 3, 4, 5, 6, 7],
]


def get_random_arr(seed, n):
    rng = np.random.RandomState(seed)
    x = rng.rand(n)
    order = np.argsort(x)
    return x, x[order], order


def test_undo_argsort_random():
    for seed in range(20):
        x, x_sorted, order = get_random_arr(seed, n=100)
        assert_array_equal(x, undo_argsort(x_sorted, order))


@pytest.mark.parametrize("method", METHODS)
def test_cluster(method):
    for k, solution in zip(range(1, len(ARR) + 1), SOLUTIONS):
        result = cluster(ARR.copy(), k, method)
        assert_array_equal(solution, result, f"k={k} method={method}")


@pytest.mark.parametrize("method", METHODS)
def test_cluster_unsorted(method):
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(ARR))
    for k, solution in zip(range(1, len(ARR) + 1), SOLUTIONS):
        result = cluster(ARR[perm], k, method)
        assert_array_equal(np.asarray(solution)[perm], result)


@pytest.mark.parametrize("method", METHODS[:2])
def test_kwargs(method):
    for k, solution in zip(range(1, len(ARR) + 1), SOLUTIONS):
        result = cluster(ARR.copy(), k, method, max_iter=200, epsilon=1e-10)
        assert_array_equal(solution, result)
    # a single iteration is still a valid labelling
    r = cluster(ARR.copy(), 3, method, max_iter=1, epsilon=1e-3)
    assert r.shape == ARR.shape


def test_invalid():
    with pytest.raises(AssertionError):
        cluster(ARR, 3, "nope")
    with pytest.raises(AssertionError):
        cluster(ARR, 0)
    with pytest.raises(AssertionError):
        cluster(ARR, len(ARR) + 1)


def test_repeated():
    arr = np.array([1, 1, 1, 1.1, 5, 5, 5])
    for method in METHODS:
        assert_array_equal(cluster(arr, 2, method), [0, 0, 0, 0, 1, 1, 1])
        assert_array_equal(cluster(arr, 3, method), [0, 0, 0, 1, 2, 2, 2])
