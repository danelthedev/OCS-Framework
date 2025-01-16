import numpy as np
from Remade.L1 import GeneralFunction, sphere_function


class PopulationV3Adaptive:
    def __init__(self, obj_function, lower_bounds, upper_bounds, population_size, max_iter, alpha_initial):
        self.obj_function = obj_function
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha_initial = alpha_initial
        self.population = self.initialize_population()
        self.convergence_history = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, len(self.lower_bounds)))

    def enforce_bounds(self, x):
        return np.clip(x, self.lower_bounds, self.upper_bounds)

    def evaluate_population(self, population):
        return np.array([self.obj_function(ind) for ind in population])

    def run(self):
        best_fitness = float('inf')
        nfe = 0  # Track function evaluations
        
        while nfe < 1000:  # Your NEF condition
            alpha = self.alpha_initial * (1 - nfe/1000)  # Less aggressive decay
            
            # Generate and evaluate new solutions
            new_population = []
            for agent in self.population:
                new_agent = self.enforce_bounds(agent + alpha * np.random.uniform(-1, 1, len(agent)))
                new_population.append(new_agent)
                nfe += 1
                if nfe >= 1000:
                    break
                
            new_population = np.array(new_population)
            new_population_fitness = self.evaluate_population(new_population)
            current_population_fitness = self.evaluate_population(self.population)
            
            # Combine and select best solutions
            combined_population = np.vstack([self.population, new_population])
            combined_fitness = np.hstack([current_population_fitness, new_population_fitness])
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            self.population = combined_population[best_indices]
            
            # Track best fitness
            current_best = np.min(combined_fitness)
            best_fitness = min(best_fitness, current_best)
            self.convergence_history.append(best_fitness)

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
