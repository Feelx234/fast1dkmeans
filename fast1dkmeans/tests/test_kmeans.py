
import unittest
import numpy as np
from fast1dkmeans.kmeans import cluster_xi, cluster_xi_space

class TestKMeans(unittest.TestCase):
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

    def test_cluster_xi(self):
        for k, solution in zip(range(1, len(self.arr)+1), self.solutions):
            result = cluster_xi(self.arr, k)
            np.testing.assert_array_equal(solution, result, f"k={k}")

    def test_cluster_xi_space(self):
        for k, solution in zip(range(1, len(self.arr)+1), self.solutions):
            result = cluster_xi_space(self.arr, k)
            np.testing.assert_array_equal(solution, result, f"k={k}")


from fast1dkmeans.tests.utils_for_test import remove_from_class, restore_to_class  # pylint: disable=wrong-import-position
class TestCommonNonCompiled(TestKMeans):
    def setUp(self):
        self.cleanup = remove_from_class(self.__class__.__bases__[0], allowed_packages=["fast1dkmeans"])

    def tearDown(self) -> None:
        restore_to_class(self.cleanup)

if __name__ == '__main__':
    unittest.main()