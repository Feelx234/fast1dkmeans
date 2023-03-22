import numpy as np
from numba import njit, float64, int64
from numba.experimental import jitclass

USE_CACHE=True

@njit([(float64[:],)], cache=USE_CACHE)
def calc_cumsum(v):
    cumsum = np.empty(len(v)+1, dtype=np.float64)
    cumsum[0]=0
    cumsum[1:] = np.cumsum(v)
    return cumsum

@njit([(float64[:],)], cache=USE_CACHE)
def calc_cumsum2(v):
    cumsum2 = np.empty(len(v)+1, dtype=np.float64)
    cumsum2[0]=0
    cumsum2[1:] = np.cumsum(np.square(v))
    return cumsum2



@njit([(float64[:], float64[:], int64, int64)], cache=USE_CACHE)
def calc_objective(cumsum, cumsum2, i, j):
    if j <= i:
        return 0.0
#            raise ValueError("j should never be larger than i")
    mu = (cumsum[j+1]-cumsum[i])/(j-i+1)
    result = cumsum2[j + 1] - cumsum2[i]
    result += (j - i + 1) * (mu * mu)
    result -= (2 * mu) * (cumsum[j + 1] - cumsum[i])
    return max(result, 0)



@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:])])
class CumsumCalculator:
    def __init__(self, v):
        self.cumsum = calc_cumsum(v)
        self.cumsum2 = calc_cumsum2(v)

    def calc(self, i, j):
        return calc_objective(self.cumsum, self.cumsum2, i, j)


@njit([(float64[:],)], cache=USE_CACHE)
def create_cumsum_calculator(arr): # pragma: no cover
    calculator = CumsumCalculator(arr)
    print(calculator.calc(0,1))


@njit(cache=USE_CACHE)
def cost_of_clustering(vals, res):
    """Compute the clustering cost of an explicitly given cluster assignment"""
    calc = CumsumCalculator(vals)
    last_i = 0
    last_val = res[0]
    cost = 0
    for i, val in enumerate(res):
        if val != last_val:
            cost += calc.calc(last_i, i-1)
            last_val = val
            last_i = i
    cost += calc.calc(last_i, len(vals)-1)

    return cost