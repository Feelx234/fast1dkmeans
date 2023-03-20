import numpy as np
from fast1dkmeans.regularized_kmeans import CumsumCalculator, __Wilber, calc_num_clusters, _conventional_algorithm


def linear_search(v):
    """Perform a linear search for Lambda"""
    n = len(v)
    calculator = CumsumCalculator(v)

    for lambda_mid in np.linspace(0, calculator.calc(0, len(v)-1)):
        result = __Wilber(n, calculator.cumsum, calculator.cumsum2, lambda_mid)
        k_mid = calc_num_clusters(result)

        result2, g = _conventional_algorithm(n, v, lambda_mid)#pylint:disable=unused-variable
        k_mid2 = calc_num_clusters(result2)

        print(lambda_mid, k_mid, k_mid2)
        #print(result)
        #print()
        #if 0:
        #    with np.printoptions(linewidth=200, precision=3):
        #        if k_mid2==4:
        #            print(g)
        #        if k_mid2==5:
        #            print(g)
        #            break
    return result