import numpy as np
from L1 import GeneralFunction, sphere_function

class RGA4Adaptive:
    def __init__(self, obj_function, lower_bounds, upper_bounds, population_size, pc_initial, pm_initial, max_nfe):
        self.obj_function = obj_function
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.population_size = population_size
        self.pc_initial = pc_initial
        self.pm_initial = pm_initial
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

    def select_parents(self, fitness):
        probabilities = fitness / np.sum(fitness)
        parents_idx = np.random.choice(np.arange(self.population_size), size=2, p=probabilities)
        return self.population[parents_idx]

    def crossover(self, parent1, parent2, alpha=0.1):
        child1 = parent1 + alpha * (parent2 - parent1)
        child2 = parent2 + alpha * (parent1 - parent2)
        return self.enforce_bounds(child1), self.enforce_bounds(child2)

    def mutate(self, individual, pm, sigma=1.0):
        if np.random.rand() < pm:
            individual += np.random.normal(0, sigma, size=individual.shape)
        return self.enforce_bounds(individual)

    def run(self):
        while self.nfe < self.max_nfe:
            fitness = self.evaluate_population()
            pc = self.pc_initial / (1 + self.nfe / self.max_nfe)
            pm = self.pm_initial / (1 + self.nfe / self.max_nfe)

            parent1, parent2 = self.select_parents(fitness)
            child1, child2 = self.crossover(parent1, parent2, alpha=0.1)
            child1 = self.mutate(child1, pm)
            child2 = self.mutate(child2, pm)

            child1_fitness = self.obj_function(child1)
            child2_fitness = self.obj_function(child2)
            self.nfe += 2

            worst_idx = np.argmax(fitness)
            if child1_fitness < fitness[worst_idx]:
                self.population[worst_idx] = child1
                fitness[worst_idx] = child1_fitness

            worst_idx = np.argmax(fitness)
            if child2_fitness < fitness[worst_idx]:
                self.population[worst_idx] = child2

        best_idx = np.argmin(self.evaluate_population())
        return self.population[best_idx], self.obj_function(self.population[best_idx])

# Parameters
lower_bounds = [-100] * 10
upper_bounds = [100] * 10
population_size = 20
pc_initial = 0.8
pm_initial = 0.1
max_nfe = 1000

# Using sphere_function from L1.py
optimizer = RGA4Adaptive(sphere_function, lower_bounds, upper_bounds, population_size, pc_initial, pm_initial, max_nfe)
best_solution, best_fitness = optimizer.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
