import numpy as np

class RandomSearcher():

    def __init__(self, max_iter, alfa, func, population_size, dimension):
        self.alfa = alfa
        self.max_iter = max_iter
        self.func = func
        self.dimension = dimension
        self.population = np.random.uniform(-10, 10, (population_size, dimension))  # Initial population
        self.fitness = np.array([self.func(ind) for ind in self.population])  # Fitness of the population

    def optimize_Population_V1_selfAdaptive(self):
        for iteration in range(self.max_iter):
            new_population = []
            new_fitness = []

            for i in range(self.population_size):
                parent = self.population[i]

                current_alfa = self.alfa + self.alfa * np.random.randn()

                offspring = parent + current_alfa * np.random.randn(self.dimension)

                offspring_fitness = self.func(offspring)

                new_population.append(offspring)
                new_fitness.append(offspring_fitness)

            combined_population = np.vstack((self.population, new_population))
            combined_fitness = np.hstack((self.fitness, new_fitness))

            best_indices = np.argsort(combined_fitness)[:self.population_size]
            self.population = combined_population[best_indices]
            self.fitness = combined_fitness[best_indices]

            print(f"Iteration {iteration + 1}, Best fitness: {self.fitness[0]}")

        # Return the best solution found
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]
