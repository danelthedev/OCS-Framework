import random


class CGA:

    def __init__(
        self,
        fitness_function,
        lower_bounds,
        upper_bounds,
        pc=0.8,
        pm=0.05,
        nfe=1000,
        population_size=100,
        gene_length=None,
    ):
        self.fitness_function = fitness_function
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.pc = pc
        self.pm = pm
        self.nfe = nfe
        self.population_size = population_size
        
        # Calculate gene length based on dimensions if not provided
        if gene_length is None:
            self.gene_length = 20 * len(lower_bounds)  # 20 bits per dimension
        else:
            self.gene_length = gene_length

        self.population = self.initialize_population()
        self.convergence_history = []

    def decode(self, individual):
        """
        Decode the binary chromosome into real values based on the length of bounds.
        """
        # Convert the binary individual to a string of bits
        binary_str = "".join(str(gene) for gene in individual)
        
        # Calculate how many bits to use per dimension
        bits_per_dimension = len(binary_str) // (len(self.lower_bounds))
        
        # To store the decoded real values
        real_values = []
        
        for i in range(len(self.lower_bounds)):
            # Extract the substring for current dimension
            start = i * bits_per_dimension
            end = (i + 1) * bits_per_dimension
            part = binary_str[start:end]
            
            # Convert binary to decimal
            decimal_value = int(part, 2)
            
            # Scale to range [lower_bound, upper_bound]
            lower = self.lower_bounds[i]
            upper = self.upper_bounds[i]
            real_value = lower + decimal_value * (upper - lower) / (2**bits_per_dimension - 1)
            
            real_values.append(real_value)
        
        return real_values

    def initialize_population(self):
        return [
            [random.randint(0, 1) for _ in range(self.gene_length)]
            for _ in range(self.population_size)
        ]

    def fitness(self, individual):
        return self.fitness_function(self.decode(individual))

    def selection(self):
        total_fitness = sum(self.fitness(individual) for individual in self.population)
        selection_probs = [
            self.fitness(individual) / total_fitness for individual in self.population
        ]
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
        best_fitness = float("inf")
        best_individual = None
        max_generations = self.nfe // self.population_size  # Calculate max generations

        # Initial population evaluation
        self.population.sort(key=self.fitness)
        current_best_fitness = self.fitness(self.population[0])
        best_fitness = current_best_fitness
        self.convergence_history.append(best_fitness)

        for generation in range(max_generations):
            # Sort population by fitness
            self.population.sort(key=self.fitness)
            current_best_individual = self.population[0]
            current_best_fitness = self.fitness(current_best_individual)

            # Update best found so far
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual

            # Create next generation
            self.population = self.replacement()
            evaluations += self.population_size

            # Track convergence
            self.convergence_history.append(best_fitness)

        # Ensure convergence history has consistent length (100 points)
        expected_length = 100
        if len(self.convergence_history) < expected_length:
            self.convergence_history.extend([best_fitness] * (expected_length - len(self.convergence_history)))
        elif len(self.convergence_history) > expected_length:
            self.convergence_history = self.convergence_history[:expected_length]

        return best_individual, best_fitness
