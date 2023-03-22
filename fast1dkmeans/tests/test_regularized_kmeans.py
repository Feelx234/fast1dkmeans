# pylint: disable=missing-function-docstring
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from fast1dkmeans.regularized_kmeans import calc_num_clusters, relabel_clusters, binary_search
from fast1dkmeans.regularized_kmeans import CumsumCalculator, calc_cluster_cost_implicit, conventional_algorithm, Wilber

class RegularizedBasics(unittest.TestCase):
    def test_calc_num_clusters(self):
        self.assertEqual(calc_num_clusters(np.array([0, 1, 1, 3, 4, 4, 4, 4])), 4)
        self.assertEqual(calc_num_clusters(np.array([0, 0, 1, 2, 3, 4, 4, 4])), 3)
        self.assertEqual(calc_num_clusters(np.array([0, 0, 0, 0, 0])), 1)
        self.assertEqual(calc_num_clusters(np.arange(10)), 10)

    def test_relabel_clusters(self):
        assert_array_equal(relabel_clusters(np.array([0, 1, 1, 3, 4, 4, 4, 4])), np.array([0, 1, 1, 2, 3, 3, 3, 3]))
        assert_array_equal(relabel_clusters(np.array([0, 0, 1, 2, 3, 4, 4, 4])), np.array([0, 0, 1, 1, 2, 2, 2, 2]))
        assert_array_equal(relabel_clusters(np.array([0, 0, 0, 0, 0])), np.array([0, 0, 0, 0, 0]))
        assert_array_equal(relabel_clusters(np.arange(10)), np.arange(10))

        
    def test_calc_cluster_cost_implicit(self):
        cc = CumsumCalculator(np.array([0, 1, 3, 4], dtype=float))
        self.assertEqual(calc_cluster_cost_implicit(np.array([0, 0, 2, 2]), cc.cumsum, cc.cumsum2), 1.0)

        cc = CumsumCalculator(np.array([0, 1, 3, 4, 7, 8, 9], dtype=float))
        self.assertEqual(calc_cluster_cost_implicit(np.array([0, 0, 2, 2, 4, 4, 4]), cc.cumsum, cc.cumsum2), 3.0)

def my_test_algorithm(self, algorithm):
    for lambda_, solution in self.lambda_input.items():
        result = algorithm(self.arr, lambda_)
        np.testing.assert_array_equal(solution, result, f"k={lambda_}")

class RegularizedKmeans(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kn = [(10, 100), (20, 100), (10, 1000), (100, 1000)]
        self.n_sampels = 1000

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

        self.lambda_input = {
            0.00 : [0, 1, 2, 3, 4, 5, 6, 7],
            0.05 : [0, 0, 1, 2, 3, 4, 4, 4],
            0.10 : [0, 0, 0, 2, 3, 3, 3, 3],
            0.15 : [0, 0, 0, 0, 3, 3, 3, 3],
            0.50 : [0, 0, 0, 0, 0, 0, 0, 3],
            1.00 : [0, 0, 0, 0, 0, 0, 0, 0]
        }

    

    def test_binary_search(self):
        for k, solution in zip(range(1, len(self.arr)+1), self.solutions):
            for method in [0,1]:
                result = binary_search(self.arr, k, method=method)
                np.testing.assert_array_equal(solution, result, f"k={k} method={method}")

    def test_conventional_algorithm(self):
        my_test_algorithm(self, conventional_algorithm)

    def test_wilber(self):
        my_test_algorithm(self, Wilber)

    def test_binary_search_against_kmeans1d_library(self):
        try:
            import kmeans1d # pylint: disable=import-outside-toplevel
        except ImportError:
            self.skipTest("kmeans1d not available")
            return
        for k, n in self.kn:
            for seed in range(self.n_sampels):
                k = 5
                n = 10
                np.random.seed(seed)
                arr = np.random.rand(n)
                arr.sort()

                clusters, _ = kmeans1d.cluster(arr, k)
                for method in [0,1]:
                    result = binary_search(arr, k, method=method)
                    np.testing.assert_array_equal(clusters, result, f"seed={seed} k={k} n={n} method={method}")


class RegularizedKmeansTestRepeated(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arr = np.array([1, 1, 1, 1.1, 5, 5, 5])

        self.lambda_input = {
            0.00 : [0, 0, 0, 3, 4, 4, 4],
            0.10 : [0, 0, 0, 0, 4, 4, 4],
        }
    def test_conventional_algorithm(self):
        my_test_algorithm(self, conventional_algorithm)

    def test_wilber(self):
        my_test_algorithm(self, Wilber)

from fast1dkmeans.tests.utils_for_test import remove_from_class, restore_to_class  # pylint: disable=wrong-import-position



class RegularizedKmeanshNonCompiled(RegularizedKmeans):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kn = [(10, 100), (20, 100)]
        self.n_sampels = 100

    def setUp(self):
        self.cleanup = remove_from_class(self.__class__.__bases__[0], allowed_packages=["fast1dkmeans"])
        #print(type(binary_search))
        #print(dir(binary_search))
        #print(self.cleanup)

    def tearDown(self) -> None:
        restore_to_class(self.cleanup)
#print(RegularizedKmeanshNonCompiled.__dict__)
#patch_class(RegularizedKmeanshNonCompiled)

if __name__ == '__main__':
    unittest.main()