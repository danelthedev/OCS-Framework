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
            # New population to store offspring
            new_population = []
            new_fitness = []

            for i in range(self.population_size):
                parent = self.population[i]

                # Self-adaptive alfa: mutate more if fitness is poor (alfa scales with inverse fitness)
                current_alfa = self.alfa / (1 + self.fitness[i])  # The higher the fitness, the smaller the step size

                # Generate offspring by adding Gaussian noise with stddev = current_alfa
                offspring = parent + current_alfa * np.random.randn(self.dimension)

                # Ensure offspring stays within bounds (-10, 10 for example)
                offspring = np.clip(offspring, -10, 10)

                # Calculate fitness of the offspring
                offspring_fitness = self.func(offspring)

                # Store the offspring and its fitness
                new_population.append(offspring)
                new_fitness.append(offspring_fitness)

            # Combine old and new population
            combined_population = np.vstack((self.population, new_population))
            combined_fitness = np.hstack((self.fitness, new_fitness))

            # Select the best agents to maintain a fixed population size
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            self.population = combined_population[best_indices]
            self.fitness = combined_fitness[best_indices]

            # Optional: print current best fitness every iteration
            print(f"Iteration {iteration + 1}, Best fitness: {self.fitness[0]}")

        # Return the best solution found
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]
