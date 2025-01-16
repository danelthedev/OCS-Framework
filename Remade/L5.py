import numpy as np
from Remade.L1 import GeneralFunction, sphere_function

class DEBest2Exp:
    def __init__(self, obj_function, lower_bounds, upper_bounds, population_size, F, CR, max_nfe):
        self.obj_function = obj_function
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.max_nfe = max_nfe
        self.population = self.initialize_population()
        self.nfe = 0

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, len(self.lower_bounds)))

    def enforce_bounds(self, x):
        return np.clip(x, self.lower_bounds, self.upper_bounds)

    def evaluate_population(self):
        fitness = np.array([self.obj_function(ind) for ind in self.population])
        self.nfe += len(self.population)
        return fitness

    def mutation(self, best, r1, r2, r3, r4):
        return best + self.F * (r1 - r2) + self.F * (r3 - r4)

    def exponential_crossover(self, target, donor):
        trial = target.copy()
        n = len(target)
        idx = np.random.randint(0, n)
        L = 0
        while L < n and np.random.rand() < self.CR:
            trial[idx] = donor[idx]
            idx = (idx + 1) % n
            L += 1
        return trial

    def selection(self, target, trial):
        if self.obj_function(trial) < self.obj_function(target):
            return trial
        return target

    def run(self):
        best_so_far = float('inf')
        self.convergence_history = [] 

        while self.nfe < self.max_nfe:
            fitness = self.evaluate_population()
            best_idx = np.argmin(fitness)
            best = self.population[best_idx]
            
            current_best = np.min(fitness)
            best_so_far = min(best_so_far, current_best)
            self.convergence_history.append(best_so_far)

            new_population = []
            for i, target in enumerate(self.population):
                r = np.random.choice([j for j in range(self.population_size) if j != i], 4, replace=False)
                r1, r2, r3, r4 = self.population[r]
                donor = self.mutation(best, r1, r2, r3, r4)
                donor = self.enforce_bounds(donor)
                trial = self.exponential_crossover(target, donor)
                trial = self.enforce_bounds(trial)
                new_population.append(self.selection(target, trial))

            self.population = np.array(new_population)
            
        expected_length = 100
        if len(self.convergence_history) < expected_length:
            self.convergence_history.extend([best_so_far] * (expected_length - len(self.convergence_history)))
        elif len(self.convergence_history) > expected_length:
            self.convergence_history = self.convergence_history[:expected_length]

        best_idx = np.argmin(self.evaluate_population())
        return self.population[best_idx], self.obj_function(self.population[best_idx])

lower_bounds = [-100] * 10
upper_bounds = [100] * 10
population_size = 20
F = 0.8
CR = 0.9
max_nfe = 1000

optimizer = DEBest2Exp(sphere_function, lower_bounds, upper_bounds, population_size, F, CR, max_nfe)
best_solution, best_fitness = optimizer.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
