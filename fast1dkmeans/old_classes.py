import numpy as np
from numba.experimental import jitclass
from numba import float32, float64, int64
from numba import typeof
from numba import njit
from fast1dkmeans.smawk import _smawk


@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:])])
class CumsumCalculator:
    def __init__(self, v):
        self.cumsum = np.empty(len(v)+1)
        self.cumsum[0]=0
        self.cumsum[1:] = np.cumsum(v)
        self.cumsum2 = np.empty(len(v)+1)
        self.cumsum2[0]=0
        self.cumsum2[1:] = np.cumsum(np.square(v))


    def calc(self, i, j):
        if j < i:
            return 0.0
        if i==j:
            return 0.0
#            raise ValueError("j should never be larger than i")
        mu = (self.cumsum[j+1]-self.cumsum[i])/(j-i+1)
        result = self.cumsum2[j + 1] - self.cumsum2[i]
        result += (j - i + 1) * (mu * mu)
        result -= (2 * mu) * (self.cumsum[j + 1] - self.cumsum[i])
        return result


@jitclass([('calculator', CumsumCalculator.class_type.instance_type), ('lambda_', float64)])
class LambdaCalculator:
    def __init__(self, calculator, lambda_):
        self.calculator = calculator
        self.lambda_ = lambda_

    def calc(self, i, j):
        return self.calculator.calc(i, j) + self.lambda_


@jitclass([('calculator', LambdaCalculator.class_type.instance_type), ("F_vals", float64[:])])
class WilberCalculator:
    def __init__(self, calculator, F_vals):
        self.calculator = calculator
        self.F_vals = F_vals

    def calc(self, j, i):
        if j<i:
            #print(i, j, np.inf)
            return np.inf
        #print(i, j, self.calculator.calc(i, j) + self.F_vals[i])
        return self.calculator.calc(i, j) + self.F_vals[i]