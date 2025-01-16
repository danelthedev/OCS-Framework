import numpy as np
from typing import Callable, Tuple


class RGA_1_Adaptive:
    def __init__(
            self,
            bounds: list[Tuple[float, float]],
            pop_size: int = 50,
            pc_start: float = 0.9,  # Initial crossover probability
            pc_end: float = 0.6,  # Final crossover probability
            pm_start: float = 0.1,  # Initial mutation probability
            pm_end: float = 0.01,  # Final mutation probability
            nfe: int = 1000,  # Number of function evaluations
    ):
        self.bounds = bounds
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.pc_start = pc_start
        self.pc_end = pc_end
        self.pm_start = pm_start
        self.pm_end = pm_end
        self.nfe = nfe

        # Calculate maximum iterations for adaptive parameters
        self.max_iter = nfe // pop_size

        self.history = []

    def get_adaptive_params(self, current_iter: int) -> Tuple[float, float]:
        """Calculate adaptive pc and pm based on current iteration"""
        progress = current_iter / self.max_iter

        # Linear interpolation between start and end values
        pc = self.pc_start + (self.pc_end - self.pc_start) * progress
        pm = self.pm_start + (self.pm_end - self.pm_start) * progress

        return pc, pm

    def initialize_population(self) -> np.ndarray:
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.dim):
            population[:, i] = np.random.uniform(
                self.bounds[i][0], self.bounds[i][1], self.pop_size
            )
        return population

    def linear_crossover(
            self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linear crossover operator"""
        # Create three offspring with modified weights
        alpha = np.random.uniform(-0.25, 1.25)  # Wider range for exploration
        
        offspring1 = 0.5 * ((1 + alpha) * parent1 + (1 - alpha) * parent2)
        offspring2 = 0.5 * ((1 - alpha) * parent1 + (1 + alpha) * parent2)
        # More explorative third offspring
        offspring3 = alpha * parent1 + (1 - alpha) * parent2

        # Stack offspring for evaluation
        offspring = np.vstack([offspring1, offspring2, offspring3])

        # Clip values to respect bounds
        for i in range(self.dim):
            offspring[:, i] = np.clip(
                offspring[:, i], self.bounds[i][0], self.bounds[i][1]
            )

        # Return the two best offspring based on their average values
        avg_values = np.mean(offspring, axis=1)
        best_indices = np.argsort(avg_values)[-2:]
        return offspring[best_indices[0]], offspring[best_indices[1]]

    def non_uniform_mutation(
            self, individual: np.ndarray, current_iter: int, pm: float
    ) -> np.ndarray:
        """Non-uniform mutation operator"""
        mutated = individual.copy()

        # Calculate adaptive mutation strength
        b = 3.0  # Reduced b for slower decay of mutation strength
        r = current_iter / self.max_iter
        mutation_strength = (1 - r**2) ** b  # Modified mutation strength calculation

        for i in range(self.dim):
            if np.random.random() < pm:
                # Randomly choose upper or lower bound direction
                if np.random.random() < 0.5:
                    delta = (self.bounds[i][1] - mutated[i]) * mutation_strength
                else:
                    delta = (mutated[i] - self.bounds[i][0]) * mutation_strength

                mutated[i] += np.random.random() * delta * (1 if np.random.random() < 0.5 else -1)
                mutated[i] = np.clip(mutated[i], self.bounds[i][0], self.bounds[i][1])

        return mutated

    def select_parent(self, fitness_values: np.ndarray) -> np.ndarray:
        # Roulette wheel selection
        fitness_values = np.where(
            fitness_values < 0, 0, fitness_values
        )  # Handle negative fitness
        total_fitness = np.sum(fitness_values)
        if total_fitness == 0:
            probs = np.ones(len(fitness_values)) / len(fitness_values)
        else:
            probs = fitness_values / total_fitness
        return self.population[np.random.choice(len(self.population), p=probs)]

    def optimize(self, objective_func: Callable) -> Tuple[np.ndarray, float, list]:
        self.population = self.initialize_population()
        evaluations = 0
        best_solution = None
        best_fitness = float("-inf")
        current_iter = 0

        while evaluations < self.nfe:
            # Get adaptive parameters for current iteration
            pc, pm = self.get_adaptive_params(current_iter)

            # Evaluate current population
            fitness_values = np.array([objective_func(ind) for ind in self.population])
            evaluations += self.pop_size

            # Update best solution
            current_best_idx = np.argmax(fitness_values)
            if fitness_values[current_best_idx] > best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_solution = self.population[current_best_idx].copy()

            # Create new population
            new_population = []

            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.select_parent(fitness_values)
                parent2 = self.select_parent(fitness_values)

                # Crossover
                if np.random.random() < pc:
                    offspring1, offspring2 = self.linear_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()

                # Mutation
                offspring1 = self.non_uniform_mutation(offspring1, current_iter, pm)
                offspring2 = self.non_uniform_mutation(offspring2, current_iter, pm)

                new_population.extend([offspring1, offspring2])

            # Trim population if odd
            self.population = np.array(new_population[: self.pop_size])
            current_iter += 1

            self.history.append(best_fitness)

        return best_solution, best_fitness, self.history