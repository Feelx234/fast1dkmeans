import numpy as np

def is_monge(M):
    """Checks whether matrix M is Monge"""
    m,n = M.shape
    for i in range(m-1):
        for j in range(n-1):
            if M[i,j]+M[i+1, j+1] > M[i, j+1] + M[i+1,j]:
                return False
    return True


def _random_monge(m, n, rands):
    """Every Monge array (including the geometric example above) is a positive linear combination of
    row-constant, column-constant, and upper-right block arrays. (This characterization was proved
    independently by Rudolf and Woeginger in 1995, Bein and Pathak in 1990, Burdyok and Trofimov
    in 1976, and possibly others.)"""
    row_const = np.repeat(rands[-m-n:-n].reshape(m,1),repeats=n, axis=1)
    col_const = np.repeat(rands[-n:].reshape(1,n),repeats=m, axis=0)
    arr = row_const+col_const
    for i in range(m):
        for j in range(n):
            arr[i:, j:] += rands[i*m+n]
    return np.flip(arr, axis=0)


def random_int_monge(m, n, block_max_val, row_max_val, col_max_val):
    """ Generates a random monge array with integer values"""
    rands= np.empty(m*n+m+n, dtype=int)
    rands[:m*n] = np.random.randint(block_max_val, size=m*n)
    rands[-m-n:-n] = np.random.randint(row_max_val, size=m)
    rands[-n:] = np.random.randint(col_max_val, size=n)
    return _random_monge(m, n, rands)