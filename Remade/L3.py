import numpy as np
from Remade.L1 import GeneralFunction, sphere_function

class CGAAdaptiveV1Replacement:
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
        self.convergence_history = []

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

    def crossover(self, parent1, parent2, pc):
        if np.random.rand() < pc:
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return self.enforce_bounds(child1), self.enforce_bounds(child2)
        return parent1, parent2

    def mutate(self, individual, pm):
        for i in range(len(individual)):
            if np.random.rand() < pm:
                individual[i] += np.random.uniform(-1, 1)
        return self.enforce_bounds(individual)

    def replace_worst(self, fitness, child):
        worst_idx = np.argmax(fitness)
        self.population[worst_idx] = child

    def run(self):
        best_so_far = float('inf')
        iteration = 0
    
        initial_fitness = self.evaluate_population()
        best_so_far = min(best_so_far, np.min(initial_fitness))
        self.convergence_history = [best_so_far]
    
        remaining_nfe = self.max_nfe - self.population_size 
        evaluations_per_iter = 2
        max_iterations = remaining_nfe // evaluations_per_iter
        
        for iteration in range(max_iterations):
            fitness = self.evaluate_population()
            pc = self.pc_initial / (1 + iteration / max_iterations)
            pm = self.pm_initial / (1 + iteration / max_iterations)

            parent1, parent2 = self.select_parents(fitness)
            child1, child2 = self.crossover(parent1, parent2, pc)
            child1 = self.mutate(child1, pm)
            child2 = self.mutate(child2, pm)

            child1_fitness = self.obj_function(child1)
            child2_fitness = self.obj_function(child2)
            self.nfe += 2

            if child1_fitness < np.max(fitness):
                self.replace_worst(fitness, child1)

            if child2_fitness < np.max(fitness):
                self.replace_worst(fitness, child2)

            current_best = np.min(fitness)
            best_so_far = min(best_so_far, current_best)
            self.convergence_history.append(best_so_far)
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
pc_initial = 0.8
pm_initial = 0.1
max_nfe = 1000

optimizer = CGAAdaptiveV1Replacement(sphere_function, lower_bounds, upper_bounds, population_size, pc_initial, pm_initial, max_nfe)
best_solution, best_fitness = optimizer.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
