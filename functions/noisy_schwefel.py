import numpy as np
from functions.base_function import BaseFunction


class NoisySchwefel(BaseFunction):
    @staticmethod
    def func(x):
        x = np.asarray(x)
        D = len(x)

        result = 0
        for i in range(D):
            inner_sum = np.sum(x[: i + 1])
            result += inner_sum**2

        noise = 1 + 0.4 * abs(np.random.normal(0, 1))

        return result * noise

    def __init__(self, x_lower=None, x_upper=None):
        x_lower = -100 if x_lower is None else x_lower
        x_upper = 100 if x_upper is None else x_upper
        super().__init__(self.func, x_lower, x_upper)
