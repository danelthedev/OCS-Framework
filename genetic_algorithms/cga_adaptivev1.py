import numpy as np


class CGAAdaptiveV1:
    def __init__(self, pop_size=50, initial_pc=0.8, initial_pm=0.1, nfe_max=1000, bits_per_var=16):
        self.pop_size = pop_size
        self.pc = initial_pc
        self.pm = initial_pm
        self.nfe_max = nfe_max
        self.bits_per_var = bits_per_var
        self.nfe = 0
        self.convergence_history = []

    def binary_to_real(self, binary, x_lower, x_upper):
        # Convert binary string to decimal
        decimal = np.zeros(len(x_lower))
        bits_per_chunk = self.bits_per_var
        max_val = 2**bits_per_chunk - 1
        
        for i in range(len(x_lower)):
            start = i * bits_per_chunk
            end = (i + 1) * bits_per_chunk
            chunk = binary[start:end]
            decimal_val = sum([bit * (2**idx) for idx, bit in enumerate(reversed(chunk))])
            # Map to real range
            decimal[i] = x_lower[i] + (decimal_val / max_val) * (x_upper[i] - x_lower[i])
        return decimal

    def real_to_binary(self, real, x_lower, x_upper):
        binary = []
        max_val = 2**self.bits_per_var - 1
        
        for i, val in enumerate(real):
            # Normalize to [0,1] then scale to integer
            normalized = (val - x_lower[i]) / (x_upper[i] - x_lower[i])
            decimal = int(normalized * max_val)
            # Convert to binary
            binary_str = format(decimal, f'0{self.bits_per_var}b')
            binary.extend([int(b) for b in binary_str])
        return np.array(binary)

    def initialize_population(self, x_lower, x_upper):
        n_vars = len(x_lower)
        total_bits = n_vars * self.bits_per_var
        population = np.random.randint(2, size=(self.pop_size, total_bits))
        return population

    def crossover(self, parents, pc):
        if np.random.random() > pc:
            return parents
        
        # Single point crossover
        point = np.random.randint(1, len(parents[0]))
        offspring1 = np.concatenate([parents[0][:point], parents[1][point:]])
        offspring2 = np.concatenate([parents[1][:point], parents[0][point:]])
        return np.array([offspring1, offspring2])

    def mutation(self, individual, pm, x_lower, x_upper, mutation_strength):
        mask = np.random.random(len(individual)) < pm
        # Bit flip mutation
        individual[mask] = 1 - individual[mask]
        return individual

    def selection(self, population, fitness):
        fitness_shifted = fitness.max() - fitness + 1e-10
        probabilities = fitness_shifted / fitness_shifted.sum()
        selected_indices = np.random.choice(
            len(population), 
            size=2, 
            p=probabilities,
            replace=True
        )
        return population[selected_indices]

    def optimize(self, problem):
        x_lower = problem.x_lower
        x_upper = problem.x_upper
        
        # Initialize binary population
        population = self.initialize_population(x_lower, x_upper)
        
        best_solution = None
        best_fitness = float('inf')
        
        # Calculate total iterations based on NFE
        total_iterations = self.nfe_max // self.pop_size
        current_iter = 0
        
        while self.nfe < self.nfe_max:
            # Update pc and pm based on iteration number
            progress = current_iter / total_iterations
            self.pc = self.initial_pc * (1 - 0.5 * progress)  # Decreases from initial_pc to 0.5*initial_pc
            self.pm = self.initial_pm * (1 + progress)        # Increases from initial_pm to 2*initial_pm
            
            fitness = np.array([
                problem.f(self.binary_to_real(ind, x_lower, x_upper)) 
                for ind in population
            ])
            self.nfe += len(population)
            
            # Update best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_solution = self.binary_to_real(population[min_idx], x_lower, x_upper)
            
            self.convergence_history.append(best_fitness)
            
            # Normalize convergence history to 100 points
            total_points = len(self.convergence_history)
            if total_points > 100:
                indices = np.linspace(0, total_points - 1, 100, dtype=int)
                self.convergence_history = [self.convergence_history[i] for i in indices]
            
            # Create new population
            new_population = []
            for _ in range(self.pop_size // 2):
                # Selection
                parents = self.selection(population, fitness)
                
                # Crossover
                offspring = self.crossover(parents, self.pc)
                
                # Mutation
                offspring[0] = self.mutation(offspring[0], self.pm, x_lower, x_upper, 0)
                offspring[1] = self.mutation(offspring[1], self.pm, x_lower, x_upper, 0)
                
                new_population.extend(offspring)
            
            population = np.array(new_population)
            
            current_iter += 1

        return best_solution, best_fitness
