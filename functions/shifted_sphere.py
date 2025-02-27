import numpy as np
from functions.base_function import BaseFunction


class ShiftedSphere(BaseFunction):
    @staticmethod
    def func(x):
        x = np.asarray(x)

        return np.sum(x**2)

    def __init__(self, x_lower=None, x_upper=None):
        x_lower = -100 if x_lower is None else x_lower
        x_upper = 100 if x_upper is None else x_upper
        super().__init__(self.func, x_lower, x_upper)
