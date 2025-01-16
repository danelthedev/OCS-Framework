import numpy as np


def sphere_function(x):
    return np.sum(x**2)


def schwefel_function(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def quartic_function(x):
    noise = np.random.uniform(0, 1)
    return np.sum([(i + 1) * (x[i] ** 4) for i in range(len(x))]) + noise


def ackley_function(x):
    a, b, c = 20, 0.2, 2 * np.pi
    sum1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
    sum2 = -np.exp(np.mean(np.cos(c * x)))
    return sum1 + sum2 + a + np.exp(1)


def rastrigin_function(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


class GeneralFunction:
    def __init__(self, lower_bounds, upper_bounds, func):
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.func = func

    def enforce_bounds(self, x):
        return np.clip(x, self.lower_bounds, self.upper_bounds)

    def evaluate(self, x):
        x = self.enforce_bounds(x)
        return self.func(x)


if __name__ == "__main__":
    lower_bounds = [-100] * 10
    upper_bounds = [100] * 10

    f1_instance = GeneralFunction(lower_bounds, upper_bounds, sphere_function)
    f2_instance = GeneralFunction(lower_bounds, upper_bounds, schwefel_function)
    f3_instance = GeneralFunction(lower_bounds, upper_bounds, quartic_function)
    f4_instance = GeneralFunction(lower_bounds, upper_bounds, ackley_function)
    f5_instance = GeneralFunction(lower_bounds, upper_bounds, rastrigin_function)

    x = np.random.uniform(-150, 150, 10)
    print("f1(x):", f1_instance.evaluate(x))
    print("f2(x):", f2_instance.evaluate(x))
    print("f3(x):", f3_instance.evaluate(x))
    print("f4(x):", f4_instance.evaluate(x))
    print("f5(x):", f5_instance.evaluate(x))
