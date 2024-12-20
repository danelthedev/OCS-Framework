import random

class CGA:

    def __init__(self, fitness_function, pc=0.8, pm=0.05, nfe=1000, population_size=100, gene_length=20):
        self.fitness_function = fitness_function

        self.pc = pc
        self.pm = pm
        self.nfe = nfe

        self.population_size = population_size
        self.gene_length = gene_length

        self.population = self.initialize_population()


    def decode(self, individual, n=2):
        """
        Decode the binary chromosome into `n` real values. 
        Each real value will be scaled to a specific range.
        """
        # Convert the binary individual to a string of bits
        binary_str = ''.join(str(gene) for gene in individual)
        
        # Length of the binary string for each real value
        length_per_part = len(binary_str) // n

        # To store the decoded real values
        real_values = []
        
        for i in range(n):
            # Extract the substring representing the current real value
            part = binary_str[i * length_per_part: (i + 1) * length_per_part]
            
            # Convert the binary part into a decimal (real number)
            decimal_value = int(part, 2)
            
            # Scale the decimal value to the desired range [-100, 100]
            real_value = -100 + decimal_value * (200 / (pow(2, length_per_part) - 1))
            
            # Append the decoded real value
            real_values.append(real_value)

        return real_values

    

    def initialize_population(self):
        return [[random.randint(0, 1) for _ in range(self.gene_length)] for _ in range(self.population_size)]


    def fitness(self, individual):
        return self.fitness_function(self.decode(individual))


    def selection(self):
        total_fitness = sum(self.fitness(individual) for individual in self.population)
        selection_probs = [self.fitness(individual) / total_fitness for individual in self.population]
        return random.choices(self.population, weights=selection_probs, k=2)


    def crossover(self, parent1, parent2):
        if random.random() < self.pc:
            crossover_point = random.randint(1, self.gene_length - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        else:
            return parent1, parent2


    def mutation(self, individual):
        return [gene if random.random() > self.pm else 1 - gene for gene in individual]


    def replacement(self):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self.selection()
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutation(child1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutation(child2))
        return new_population


    def run(self):
        evaluations = 0
        best_fitness = float('inf')  # Initialize to a very large value for minimization
        best_individual = None

        for generation in range(self.nfe):
            # Sort the population in ascending order of fitness (for minimization)
            self.population.sort(key=self.fitness)

            current_best_individual = self.population[0]  # Best individual (lowest fitness)
            current_best_fitness = self.fitness(current_best_individual)

            # Print progress
            print(f"Generation {generation + 1}: Best fitness = {current_best_fitness}")

            # Update the best fitness and individual if necessary
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual

            # Perform the replacement step to create the next generation
            self.population = self.replacement()

            # Check if we found an optimal solution
            evaluations += self.population_size  # Count evaluations for this generation
            if best_fitness == 0:  # Since the shifted elliptic function tends to 0 at the global minimum
                print(f"Optimal solution found at generation {generation + 1}")
                break

            # Stop if we've reached the max number of fitness evaluations
            if evaluations >= self.nfe:
                break

        return best_individual, best_fitness
