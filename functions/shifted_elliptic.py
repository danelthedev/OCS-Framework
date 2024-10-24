import numpy as np
from base_function import BaseFunction


class ShiftedElliptic(BaseFunction):
    def shifted_elliptic_function(x, o, M, f_bias=450):
        # Calculate z = (x - o) * M
        z = np.dot((x - o), M)

        # Calculate the function value
        D = len(x)
        sum_value = np.sum(
            [(10**6) ** ((i - 1) / (D - 1)) * z[i] ** 2 for i in range(D)]
        )

        # Return the function value with bias
        return sum_value + f_bias


# Example usage:
D = 3  # Dimension
o = np.zeros(D)  # Shifted global optimum
M = np.eye(D)  # Orthogonal matrix

# Define bounds
x_lower = [-100] * D
x_upper = [100] * D

# Create oracle
oracle = BaseFunction(
    lambda x: ShiftedElliptic.shifted_elliptic_function(x, o, M), x_lower, x_upper
)

# Test the oracle
print(oracle.evaluate([0, 0, 0]))
print(oracle.evaluate([-50, 50, 100]))
