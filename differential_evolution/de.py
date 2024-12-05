import numpy as np

class DE:
    def __init__(self, bounds, crossp=0.7, popsize=20, its=1000):
        self.crossp = crossp
        self.bounds = bounds
        self.popsize = popsize
        self.its = its

        self.initial_population = np.random.uniform(bounds[:, 0], bounds[:, 1], (popsize, bounds.shape[0]))



