import numpy as np
from numba import njit, float64, int64
from numba.experimental import jitclass
from fast1dkmeans.smawk_iter import _smawk_iter
from fast1dkmeans.common import calc_objective, CumsumCalculator
from fast1dkmeans.regularized_kmeans import __Wilber, relabel_clusters



@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:]), ('D', float64[:,:]), ('D_row', int64)])
class XiaolinCalculator():
    def __init__(self, cumsum, cumsum2, D):
        self.cumsum = cumsum
        self.cumsum2 = cumsum2
        self.D = D
        self.D_row=0

    def set_d_row(self, val):
        self.D_row=val

    def calc(self, i, j):
        col = i if i < j - 1 else j - 1
        return self.D[self.D_row, col] + calc_objective(self.cumsum, self.cumsum2, j, i)

@njit(cache=True)
def cluster_xi(v, k):
    """Optimal quantization by matrix searching by Xiaolin Wu"""

    cost_calculator = CumsumCalculator(v)
    n = len(v)
    D = np.empty((2, n), dtype=np.float64)
    T = np.empty((k, n), dtype=np.int64)
    T[0,:]=0
    for j in range(n):
        D[0,j] = cost_calculator.calc(0, j)
    xi_calculator = XiaolinCalculator(cost_calculator.cumsum, cost_calculator.cumsum2, D)


    n = len(v)
    row_argmins = np.empty(n, dtype=T.dtype)
    rows = np.arange(n)
    cols = np.arange(n)
    for _k in range(1, k):
        D_row = (_k-1) % 2
        xi_calculator.set_d_row(D_row)
        _smawk_iter(rows, cols, xi_calculator, row_argmins)
        T[_k,:] = row_argmins
        #print(row_argmins)
        next_d_row =  _k % 2
        for i, argmin in enumerate(row_argmins):
            min_val = xi_calculator.calc(i, argmin)
            D[next_d_row, i] = min_val
    return back_track_to_get_clustering(T, n, k)

@njit(cache=True)
def cluster_xi_space(v, k):
    """Same as cluster_xi but with space saving technique applied"""

    if k == 1:
        return np.zeros(len(v), dtype=np.int32)
    cost_calculator = CumsumCalculator(v)
    n = len(v)
    D = np.empty((2, n), dtype=np.float64)
    T = np.empty(n, dtype=np.int64)
    T[:]=0
    for j in range(n):
        D[0,j] = cost_calculator.calc(0, j)
    xi_calculator = XiaolinCalculator(cost_calculator.cumsum, cost_calculator.cumsum2, D)


    n = len(v)
    rows = np.arange(n)
    cols = np.arange(n)
    D_row = 0
    next_d_row = 0
    for _k in range(1, k+1):
        D_row = (_k-1) % 2
        xi_calculator.set_d_row(D_row)
        _smawk_iter(rows, cols, xi_calculator, T)
        #print(row_argmins)
        next_d_row =  _k % 2
        for i, argmin in enumerate(T):
            min_val = xi_calculator.calc(i, argmin)
            D[next_d_row, i] = min_val
    #print(k)
    k_plus1_row = next_d_row #(k+1) % 2
    k_row =  D_row #(k) % 2
    lambda_ =  D[k_row, n-1] - D[k_plus1_row, n-1]
    assert lambda_ >= 0
    result = __Wilber(n, xi_calculator.cumsum, xi_calculator.cumsum2, lambda_)
    return relabel_clusters(result)

@njit
def back_track_to_get_clustering(T, n, k):
    """compute cluster assignmento of n points to k clsuters from T
    T implicitly represents for each value k' <= k and n' <= n how the optimal clustering
    of n' points into k' clusters.

    To understand how the backtracking works, lets consider the recursive formulation:
    backtrack(T, n, k):
        n' = T[k, n]
        assign points n' to n to cluster_k
        if k > 0:
            # assign the remaining n' points to k-1 clusters
            backtrack(T, n', k-1, last_n=n)
    
    """
    out = np.empty(n, dtype=np.int64)
    
    start = n
    for k_ in range(k-1, -1, -1):
        stop = start
        start = T[k_, start-1]
        for i in range(start, stop): # assign points to clusters
            out[i] = k_
    return out