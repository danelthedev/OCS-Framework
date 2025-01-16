import numpy as np
from functions.base_function import BaseFunction


class SchwefelBounds(BaseFunction):
    def __init__(self, x_lower=None, x_upper=None):
        x_lower = -100 if x_lower is None else x_lower
        x_upper = 100 if x_upper is None else x_upper
        super().__init__(self.func, x_lower, x_upper)

        self.A = None
        self.B = None

    def initialize_parameters(self, D):
        self.A = np.random.randint(-500, 501, size=(D, D))
        while np.linalg.det(self.A) == 0:
            self.A = np.random.randint(-500, 501, size=(D, D))

        self.B = np.zeros(D)

    def func(self, x):
        x = np.asarray(x)
        D = len(x)

        if self.A is None or self.A.shape[0] != D:
            self.initialize_parameters(D)

        results = np.abs(np.dot(self.A, x) - self.B)

        return np.max(results)
