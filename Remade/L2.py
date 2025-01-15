import numpy as np
from L1 import GeneralFunction, sphere_function


class PopulationV3Adaptive:
    def __init__(self, obj_function, lower_bounds, upper_bounds, population_size, max_iter, alpha_initial):
        self.obj_function = obj_function
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha_initial = alpha_initial
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, len(self.lower_bounds)))

    def enforce_bounds(self, x):
        return np.clip(x, self.lower_bounds, self.upper_bounds)

    def evaluate_population(self, population):
        return np.array([self.obj_function(ind) for ind in population])

    def run(self):
        for iter_no in range(1, self.max_iter + 1):
            alpha = self.alpha_initial / iter_no
            new_population = []
            for agent in self.population:
                new_agents = [self.enforce_bounds(agent + alpha * np.random.uniform(-1, 1, len(agent))) for _ in
                              range(self.population_size)]
                new_population.extend(new_agents)
            new_population = np.array(new_population)
            new_population_fitness = self.evaluate_population(new_population)

            current_population_fitness = self.evaluate_population(self.population)
            better_agents = new_population[new_population_fitness < np.max(current_population_fitness)]
            self.population = np.vstack([self.population, better_agents])[:self.population_size]

        best_solution_idx = np.argmin(self.evaluate_population(self.population))
        return self.population[best_solution_idx], self.obj_function(self.population[best_solution_idx])


# Parameters
lower_bounds = [-100] * 10
upper_bounds = [100] * 10
population_size = 10
max_iter = 100
alpha_initial = 1.0

# Using sphere_function from L1.py
optimizer = PopulationV3Adaptive(sphere_function, lower_bounds, upper_bounds, population_size, max_iter, alpha_initial)
best_solution, best_fitness = optimizer.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
