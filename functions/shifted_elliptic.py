import numpy as np
from base_function import BaseFunction


class ShiftedElliptic(BaseFunction):
    def shifted_elliptic_function(x, o, M):
        # Calculate z = (x - o) * M
        z = np.dot((x - o), M)

        # Calculate the function value
        D = len(x)
        sum_value = np.sum(
            [(10**6) ** ((i - 1) / (D - 1)) * z[i] ** 2 for i in range(D)]
        )

        # Return the function value with bias
        return sum_value