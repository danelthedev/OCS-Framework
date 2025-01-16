import numpy as np
from functions.base_function import BaseFunction


class ShiftedElliptic(BaseFunction):

    @staticmethod
    def func(x):
        # Convert input to numpy array if not already
        x = np.asarray(x)
        D = len(x)
        
        # Note: The original formula uses i starting from 1, so we adjust the power
        sum_value = np.sum(
            [(10**6) ** (i / (D - 1)) * x[i] ** 2 for i in range(D)]
        )
        return sum_value

    def __init__(self, x_lower, x_upper):
        # Default bounds from CEC 2005: [-100, 100]^D
        x_lower = -100 if x_lower is None else x_lower
        x_upper = 100 if x_upper is None else x_upper
        super().__init__(self.func, x_lower, x_upper)
