import numpy as np
from numba import njit, float64
from numba.experimental import jitclass
USE_CACHE=False





@njit(cache=False)# cache=False !!!!!!!!
def _smawk(rows, cols, calculator, result):
    """This function can as of now NOT BE CACHED as numba cannot handle recursive functions in cache
    THIS IS ALSO TRUE FOR CALLING FUNCTIONS!
    https://github.com/numba/numba/issues/6061
    """
    #print(rows, cols)
    if len(rows)==0:
        return
    if len(cols)==1:
        for r in rows:
            result[r]=cols[0]
        return

    _cols = reduce(rows, cols, calculator)
    result[rows[0]]=_cols[0]

    #result_val[rows[0]]=lookup(rows[0], _cols[0])
    #else:
    #    _cols = cols.copy()

    odd_rows = rows[1::2]

    _smawk(odd_rows, _cols, calculator, result)
    if len(rows)>2:
        interpolate(rows, _cols, calculator, result)

def smawk_implicit(m, n, lookup):
    results = np.zeros(m, dtype=int)
    rows = np.arange(m)
    cols = np.arange(n)
    _smawk(rows, cols, lookup, results)
    return results

@njit()
def interpolate(rows, cols, calculator, result):
    curr=0
    for i in range(2, len(rows), 2):
        start_col = result[rows[i-1]]
        if i+1 < len(rows):
            stop_col = result[rows[i+1]]
        else:
            stop_col = cols[len(cols)-1]
        while cols[curr] < start_col:
            curr+=1
        best = curr
        best_val = calculator.calc(rows[i], cols[best])
        while cols[curr] < stop_col:
            tmp = calculator.calc(rows[i], cols[curr+1])
            if best_val > tmp:
                best = curr+1
                best_val = tmp

            curr+=1
        result[rows[i]]=cols[best]

@njit()
def reduce(rows, cols, calculator):
    # https://courses.engr.illinois.edu/cs473/sp2016/notes/06-sparsedynprog.pdf
    m = len(rows)
    #n = len(cols)
    S = np.empty(m+1,dtype=np.int32)
    S[0]=0
    r = 0
    for k in cols:
        if k != S[r]:
            while r >= 0:
                if calculator.calc(rows[r], S[r]) > calculator.calc(rows[r], k):
                    r-=1
                else:
                    break
        if r < m-1:
            r+=1
            S[r]=k
    return S[:r+1]

@jitclass([('A', float64[:,:])])
class ArrayCalculator:
    def __init__(self, A):
        self.A = A
    def calc(self, i, j):
        return self.A[i,j]


# @njit([(float64[:,:],)], cache=True)
# def create_array_calculator(arr): # pragma: no cover
#     calculator = ArrayCalculator(arr)
#     print(calculator.calc(0,1))

@njit(cache=False)
def smawk_array(A):
    A = A.copy()
    A = A.astype(np.float64)
    m, n = A.shape
    calculator = ArrayCalculator(A)
    results = np.zeros(m, dtype=np.int64)
    rows = np.arange(m)
    cols = np.arange(n)
    _smawk(rows, cols, calculator, results)
    return results