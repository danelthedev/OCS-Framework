from functions.base_function import BaseFunction

class ShiftedSchwefel(BaseFunction):

    def func(x):
        D = len(x)
        result = 0
        for i in range(1, D+1):
            inner_sum = sum(x[j] for j in range(i))
            result += inner_sum ** 2
        return result

    def __init__(self, x_lower, x_upper):
        super().__init__(x_lower, x_upper)
        self.f = self.func