import numpy as np
from functions.base_function import BaseFunction


class ShiftedElliptic(BaseFunction):

    @staticmethod
    def func(x):
        D = len(x)
        sum_value = np.sum(
            [(10**6) ** ((i - 1) / (D - 1)) * x[i] ** 2 for i in range(D)]
        )
        return sum_value

    def __init__(self, x_lower, x_upper):
        super().__init__(self.func, x_lower, x_upper)
