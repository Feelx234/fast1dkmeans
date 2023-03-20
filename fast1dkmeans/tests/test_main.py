# pylint: disable=missing-function-docstring
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from fast1dkmeans.main import undo_argsort, undo_argsort_numba, cluster

def get_random_arr(seed, n):
    np.random.seed(seed)
    x = np.random.rand(100)
    order = np.argsort(x)
    x_sorted = x[order]
    return x, x_sorted, order


class RegularizedKmeans(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.array([0.01864729, 0.23297427, 0.38786064, 0.56103022, 0.74712164, 0.80063267, 0.8071052, 0.86354185])
        self.solutions = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 2, 2, 2, 2],
            [0, 1, 1, 2, 3, 3, 3, 3],
            [0, 1, 2, 3, 4, 4, 4, 4],
            [0, 1, 2, 3, 4, 4, 4, 5],
            [0, 1, 2, 3, 4, 5, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ]

    def test_undo_argsort_random(self):
        for seed in range(20):
            x, x_sorted, order = get_random_arr(seed, n=100)
            x_undone = undo_argsort(x_sorted, order)
            assert_array_equal(x, x_undone)

    def test_undo_argsort_numba_random(self):
        for seed in range(20):
            x, x_sorted, order = get_random_arr(seed, n=100)
            x_undone2 = undo_argsort_numba(x_sorted, order)
            assert_array_equal(x, x_undone2)

    def test_cluster(self):
        for k, solution in zip(range(1, len(self.arr)+1), self.solutions):
            for method in ("binary-search-interpolation",
                            "binary-search-normal",
                            "dynamic-programming-kn",
                            "dynamic-programming-space",
                            "dynamic-programming"):
                result = cluster(self.arr.copy(), k, method)
                np.testing.assert_array_equal(solution, result, f"k={k} method={method}")


if __name__ == '__main__':
    unittest.main()