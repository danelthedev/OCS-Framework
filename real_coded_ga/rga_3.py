import numpy as np
from typing import Callable, Tuple


class RGA_3:
    def __init__(
        self,
        bounds: list[Tuple[float, float]],
        pop_size: int = 50,
        pc: float = 0.8,  # Fixed crossover probability
        pm: float = 0.1,  # Fixed mutation probability
        nfe: int = 1000,  # Number of function evaluations
    ):
        self.bounds = bounds
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.nfe = nfe

    def initialize_population(self) -> np.ndarray:
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.dim):
            population[:, i] = np.random.uniform(
                self.bounds[i][0], self.bounds[i][1], self.pop_size
            )
        return population

    def blx_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha = 0.5  # BLX-0.5 as specified

        # Calculate bounds for each dimension
        min_vals = np.minimum(parent1, parent2)
        max_vals = np.maximum(parent1, parent2)
        diff = np.abs(parent1 - parent2)

        # Extended bounds according to BLX-α formula
        lower_bounds = min_vals - alpha * diff
        upper_bounds = max_vals + alpha * diff

        # Generate two children
        child1 = np.random.uniform(lower_bounds, upper_bounds)
        child2 = np.random.uniform(lower_bounds, upper_bounds)

        # Clip values to respect bounds
        for i in range(self.dim):
            child1[i] = np.clip(child1[i], self.bounds[i][0], self.bounds[i][1])
            child2[i] = np.clip(child2[i], self.bounds[i][0], self.bounds[i][1])

        return child1, child2

    def random_mutation(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        for i in range(self.dim):
            if np.random.random() < self.pm:
                mutated[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
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

    def optimize(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        self.population = self.initialize_population()
        evaluations = 0
        best_solution = None
        best_fitness = float("-inf")

        while evaluations < self.nfe:
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
                if np.random.random() < self.pc:
                    offspring1, offspring2 = self.blx_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()

                # Mutation
                offspring1 = self.random_mutation(offspring1)
                offspring2 = self.random_mutation(offspring2)

                new_population.extend([offspring1, offspring2])

            # Trim population if odd
            self.population = np.array(new_population[: self.pop_size])

        return best_solution, best_fitness
