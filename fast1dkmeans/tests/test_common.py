
import unittest
import numpy as np
from fast1dkmeans.common import cost_of_clustering

class TestCommon(unittest.TestCase):
    def test_cost_of_clustering(self):
        self.assertEqual(cost_of_clustering(np.array([0, 1, 3, 4, 7, 8, 9], dtype=float), np.array([0,0, 1,1, 2, 2, 2], dtype=np.int64)), 3)


from fast1dkmeans.tests.utils_for_test import remove_from_class, restore_to_class # pylint: disable=wrong-import-position
class TestCommonNonCompiled(TestCommon):
    def setUp(self):
        self.cleanup = remove_from_class(self.__class__.__bases__[0], allowed_packages=["fast1dkmeans"])

    def tearDown(self) -> None:
        restore_to_class(self.cleanup)

if __name__ == '__main__':
    unittest.main()