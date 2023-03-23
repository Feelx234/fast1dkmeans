import numpy as np
from numba import njit, float64
from numba.experimental import jitclass
from fast1dkmeans.smawk import interpolate



@njit()
def reduce_iter(rows, cols, calculator, col_buffer):
    # https://courses.engr.illinois.edu/cs473/sp2016/notes/06-sparsedynprog.pdf

    m = len(rows)
    S = col_buffer
    S[0]=cols[0]
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
    return r+1

@njit
def calc_max_col_space(n_rows, n_cols):
    max_cols = n_cols
    depth=0
    col_starts = [0, n_cols]
    while True:
        step_size = 2**depth
        val = min(n_rows//step_size, n_cols)
        max_cols+=val
        col_starts.append(max_cols)
        depth+=1
        if val == 0:
            break
    return np.array(col_starts), depth

@njit()
def _smawk_iter(rows_in, cols_in, calculator, result):
    col_starts, max_depth = calc_max_col_space(len(rows_in), len(cols_in))
    col_ends = col_starts[1:].copy()
    #col_buffer= np.empty(col_starts[-1], dtype=cols_in.dtype)
    col_buffer= - np.ones(col_starts[-1], dtype=cols_in.dtype)
    col_buffer[:len(cols_in)]=cols_in
    #print(col_starts)
    #print(col_buffer)
    #print(max_depth)

    depth = 0
    for depth in range(0, max_depth):
        step_size = 2**depth
        rows = rows_in[step_size-1::step_size]
        cols = col_buffer[col_starts[depth]:col_ends[depth]]
        #print(depth, rows, cols)

        if len(rows)==0:
            break
        if len(cols)==1:
            for r in rows:
                result[r]=cols[0]
            break
        S = col_buffer[col_starts[depth+1]:col_ends[depth+1]]
        #print(S)
        col_ends[depth+1] = col_starts[depth+1]+reduce_iter(rows, cols, calculator, S)

        _cols = col_buffer[col_starts[depth+1]:col_ends[depth+1]]
        #print(_cols)
        #print()
        result[rows[0]]=_cols[0]
    #print(col_buffer)
    #print("fill")
    for depth in range(max_depth, -1, -1):
        step_size = 2**depth
        rows = rows_in[step_size-1::step_size]
        #print(rows)
        if len(rows)==0:
            continue
        _cols = col_buffer[col_starts[depth+1]:col_ends[depth+1]]
        #print(cols, "\n")
        if len(rows)>2:
            interpolate(rows, _cols, calculator, result)
    #print(result)


@jitclass([('A', float64[:,:])])
class ArrayCalculator:
    def __init__(self, A):
        self.A = A
    def calc(self, i, j):
        return self.A[i,j]

@njit(cache=True)
def smawk_iter_array(A):
    A = A.copy()
    A = A.astype(np.float64)
    m, n = A.shape
    calculator = ArrayCalculator(A)
    results = np.zeros(m, dtype=np.int64)
    rows = np.arange(m)
    cols = np.arange(n)
    _smawk_iter(rows, cols, calculator, results)
    return results