import numpy as np
from numba.experimental import jitclass
from numba import float64, int64
from numba import njit
from fast1dkmeans.smawk_iter import _smawk_iter

from fast1dkmeans.common import calc_cumsum, calc_cumsum2, calc_objective, CumsumCalculator

USE_CACHE=True





@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:]), ('lambda_', float64)])
class LambdaCalculator:
    def __init__(self, v, lambda_):
        self.cumsum = calc_cumsum(v)
        self.cumsum2 = calc_cumsum2(v)
        self.lambda_ = lambda_

    def calc(self, i, j):
        return calc_objective(self.cumsum, self.cumsum2, i, j) + self.lambda_



@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:]), ('lambda_', float64), ("F_vals", float64[:])])
class WilberCalculator:
    def __init__(self, cumsum, cumsum2, lambda_, F_vals):
        self.cumsum = cumsum
        self.cumsum2 = cumsum2
        self.lambda_ = lambda_
        self.F_vals = F_vals

    def calc(self, j, i): # i <-> j interchanged is not a bug!
        if j<i:
            #print(i, j, np.inf)
            return np.inf
        #print(i, j, self.calculator.calc(i, j) + self.F_vals[i])
        return calc_objective(self.cumsum, self.cumsum2, i, j) + self.lambda_ + self.F_vals[i]










@njit([(float64[:], float64)], cache=USE_CACHE)
def create_lambda_calculator(arr, lambda_): # pragma: no cover
    calculator = LambdaCalculator(arr, lambda_)
    print(calculator.calc(0,1))



@njit([(float64[:], float64)], cache=USE_CACHE)
def create_wilber_calculator(arr, lambda_): # pragma: no cover
    calculator = WilberCalculator(arr, arr, lambda_, arr)
    print(calculator.calc(0,1))






@njit([(int64, float64[:], float64[:], float64)], cache=USE_CACHE)
def __Wilber(n, cumsum, cumsum2, lambda_):
    """Solves the REGULARIZED 1d kmeans problem in O(n)
    this is an implementation of the proposed algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    F = np.empty(n, dtype=np.int32)
    F_vals = np.empty(n+1, dtype=np.float64)
    H = np.empty(n, dtype=np.int32)
    H_vals = np.empty(n+1, dtype=np.float64)
    F_vals[0]=0
    F[0]=100000 # never acessed
    c = 0
    r = 0
    wil_calculator = WilberCalculator(cumsum, cumsum2, lambda_, F_vals)
    while c < n:
        p = min(2*c-r+1, n)
        #print("p", p)
        #print("F_input", r, c+1, c, p)
        _smawk_iter(np.arange(c, p), np.arange(r, c+1), wil_calculator, F)
        #print("F", F)

        for j in range(c, p):
            F_vals[j+1] = wil_calculator.calc(j, F[j])
        #print("F_val", F_vals)
        #print("H", c+1, p, c+1,p)
        _smawk_iter(np.arange(c+1, p), np.arange(c+1, p), wil_calculator, H)
        for j in range(c+1, p):
            H_vals[j+1] = wil_calculator.calc(j, H[j])
        #print("H_val", H_vals)
        #print()
        j0=p+1
        for j in range(c+2, p+1):
            #print("<< j",j, H_vals[j], F_vals[j])
            if H_vals[j] < F_vals[j]:
                F[j-1] = H[j-1]
                j0 = j
                break
        if j0==p+1:
            c = p
        else:
            #raise ValueError()
            #print(">>>>>>j0", j0)
            F_vals[j0] = H_vals[j0]
            r = c+1
            c=j0
    return F#relabel(F)

@njit([(int64, float64[:], float64)], cache=USE_CACHE)
def _Wilber(n, v, lambda_):
    cumsum = calc_cumsum(v)
    cumsum2 = calc_cumsum2(v)
    return __Wilber(n, cumsum, cumsum2, lambda_)



def Wilber(arr, lambda_):
    """Solves the REGULARIZED 1d kmeans problem in O(n)
    this is an implementation of the proposed algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    n = len(arr)
    #calculator = _create_calculator(arr, lambda_)
    return _Wilber(n, arr, lambda_)



def conventional_algorithm(vals, lambda_):
    """Solves the REGULARIZED 1d kmeans problem in O(n^2)
    this is an implementation of the conventional algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    n = len(vals)
    F, g = _conventional_algorithm(n, vals, lambda_) #pylint: disable=unused-variable
    with np.printoptions(linewidth=200):
        pass
        #print(g)
    return F

@njit([(int64, float64[:], float64)], cache=USE_CACHE)
def _conventional_algorithm(n, vals, lambda_):
    """Solves the REGULARIZED 1d kmeans problem in O(n^2)
    this is an implementation of the conventional algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    calculator = LambdaCalculator(vals, lambda_)
    g = np.zeros((n,n))
    F_val = np.zeros(n+1)
    F_val[0]=0
    F = np.zeros(n, dtype=np.int32)
    for j in range(1,n+1):
        for i in range(j):
            #print(i, j-1, calculator.calc(i, j-1))
            g[i,j-1] = F_val[i]+calculator.calc(i, j-1)
        F[j-1] = np.argmin(g[:j,j-1])
        F_val[j] = g[F[j-1],j-1]

    return F, g


@njit(cache=USE_CACHE)
def calc_num_clusters(result):
    """Compute the number of clusters encoded in results
    Can be used on e.g. the result of _conventional_algorithm, Weber
    """
    num_clusters = 0
    curr_pos = len(result)-1
    while result[curr_pos]>0:
        curr_pos = result[curr_pos]-1
        num_clusters+=1
    return num_clusters+1


@njit(cache=USE_CACHE)
def relabel_clusters(result):
    num_clusters = calc_num_clusters(result)-1
    out = np.empty_like(result)
    curr_pos = len(result)-1
    while result[curr_pos]>0:
        out[result[curr_pos]:curr_pos+1] = num_clusters
        curr_pos = result[curr_pos]-1
        num_clusters-=1
    out[0:curr_pos+1] = num_clusters
    return out


@njit(cache=USE_CACHE)
def calc_cluster_cost_implicit(result, cumsum, cumsum2):
    """Compute the number of clusters encoded in results
    Can be used on e.g. the result of _conventional_algorithm, Weber
    """
    cost = 0
    curr_pos = len(result)-1
    while result[curr_pos]>0:
        cost+=calc_objective(cumsum, cumsum2, result[curr_pos], curr_pos)
        curr_pos = result[curr_pos]-1
    cost+=calc_objective(cumsum, cumsum2, 0, curr_pos)
    return cost






@njit(cache=USE_CACHE)
def binary_search(v, k, max_iter=200, epsilon=1e-10, method = 0):
    """Compute the optimal k-means clustering for sorted v
    This function implements the algorithm proposed in
        "Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D"
            by Gronlund et al.

    """
    assert method in (0,1)
    if len(v)==k:
        return np.arange(len(v), dtype=np.int32)

    n = len(v)
    calculator = CumsumCalculator(v)
    #print(calculator.cumsum)
    #print(calculator.cumsum2)
    l_low = 0
    k_low = n
    c_low = 0

    l_high = calculator.calc(0, len(v)-1)
    k_high = 1
    c_high = l_high


    for _ in range(max_iter):

        lambda_mid = (l_high + l_low)/2
        if method == 1:
            lambda_test = (c_high - c_low)/(k_low - k_high)
            if lambda_test >= l_low and lambda_test <= l_high:
                lambda_mid = lambda_test
        #print("low", k_low, l_low, c_low)
        #print("high", k_high, l_high, c_high)
        result = __Wilber(n, calculator.cumsum, calculator.cumsum2, lambda_mid)
        k_mid = calc_num_clusters(result)
        #print(k_mid, lambda_mid, "\n")
        if k_mid == k:
            if l_high - l_low < epsilon:
                break
            l_high =lambda_mid
            k_high = k_mid
        elif k_mid < k:
            # to few clusters, need to decrease lambda
            l_high =lambda_mid
            k_high = k_mid
            if method==1:
                c_high = calc_cluster_cost_implicit(result, calculator.cumsum, calculator.cumsum2)
                #c_high2 = cost_of_clustering(v, relabel_clusters(result))
        else:
            # to many clusters, need to increase lambda
            l_low =lambda_mid
            k_low = k_mid
            if method==1:
                c_low = calc_cluster_cost_implicit(result, calculator.cumsum, calculator.cumsum2)
                #c_low2 = cost_of_clustering(v, relabel_clusters(result))
        #print()
    return relabel_clusters(result)