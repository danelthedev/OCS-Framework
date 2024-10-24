import numpy as np

class RandomSearcher():

    def __init__(self, max_iter, alfa, func, population_size, dimension, print_results = False):
        self.alfa = alfa
        self.max_iter = max_iter
        self.func = func
        self.dimension = dimension
        self.population_size = population_size
        self.population = np.random.uniform(-10, 10, (population_size, dimension))  # Initial population
        self.fitness = np.array([self.func(ind) for ind in self.population])  # Fitness of the population

        self.print_results = print_results

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

            if self.print_results:
                print(f"Iteration {iteration + 1}, Best fitness: {self.fitness[0]}")

        # Return the best solution found
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]
    
    def optimize_Population_V3(self):
        for iteration in range(self.max_iter):
            # New population to store offspring
            new_population = []
            new_fitness = []

            for i in range(len(self.population)):
                parent = self.population[i]

                # Generate multiple offspring for each parent
                for _ in range(self.alfa):  # Assuming alfa is the number of offspring per agent
                    offspring = parent + np.random.randn(self.dimension)

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

            # Select all better agents
            better_indices = np.where(combined_fitness < np.min(self.fitness))[0]
            if len(better_indices) > 0:
                self.population = combined_population[better_indices]
                self.fitness = combined_fitness[better_indices]
            else:
                # If no better agents, keep the current population
                self.population = combined_population[:len(self.population)]
                self.fitness = combined_fitness[:len(self.fitness)]

            #Print current best fitness every iteration
            if self.print_results:
                print(f"Iteration {iteration + 1}, Best fitness: {np.min(self.fitness)}")

        # Return the best solution found
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]
