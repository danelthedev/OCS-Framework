import numpy as np


class CGAAdaptiveV1:
    def __init__(self, pop_size=50, initial_pc=0.8, initial_pm=0.1, nfe_max=1000):
        super().__init__()
        self.pop_size = pop_size
        self.initial_pc = initial_pc
        self.initial_pm = initial_pm
        self.nfe_max = nfe_max

    def adapt_rates(self, iter_no, max_iter):
        # Non-linear adaptive rates using sigmoid function
        progress = iter_no / max_iter
        pc = self.initial_pc / (1 + np.exp(-10 * (0.5 - progress)))
        pm = self.initial_pm * (1 / (1 + np.exp(-10 * (progress - 0.5))))
        return max(0.6, min(0.9, pc)), max(0.001, min(0.1, pm))

    def selection(self, population, fitness):
        # Calculate total fitness
        total_fitness = np.sum(fitness)

        # Calculate relative fitness (probabilities)
        probs = fitness / total_fitness

        # Calculate cumulative probabilities
        cum_probs = np.cumsum(probs)

        # Select two parents using cumulative probabilities
        selected = []
        for _ in range(2):
            r = np.random.random()
            selected_idx = np.where(cum_probs >= r)[0][0]
            selected.append(population[selected_idx])

        return np.array(selected)

    def crossover(self, parents, pc):
        if np.random.random() > pc:
            return parents

        # Single-point crossover
        parent1, parent2 = parents
        point = np.random.randint(1, len(parent1))
        offspring1 = np.concatenate([parent1[:point], parent2[point:]])
        offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        return np.array([offspring1, offspring2])

    def mutation(self, individual, pm, x_lower, x_upper):
        for i in range(len(individual)):
            if np.random.random() < pm:
                # Flip bit (for binary representation)
                individual[i] = (
                    x_lower[i] if individual[i] == x_upper[i] else x_upper[i]
                )
        return individual

    def optimize(self, problem):
        nfe = 0
        D = len(problem.x_lower)
        max_iter = self.nfe_max // self.pop_size

        # Initialize population
        population = np.random.uniform(
            problem.x_lower, problem.x_upper, size=(self.pop_size, D)
        )

        best_solution = None
        best_fitness = float("inf")

        for iter_no in range(max_iter):
            # Evaluate population
            fitness = np.array([problem.f(x) for x in population])
            nfe += self.pop_size

            # Update best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_solution = population[min_idx].copy()

            # Adapt rates
            pc, pm = self.adapt_rates(iter_no, max_iter)

            # Create new population
            new_population = []
            for _ in range(self.pop_size // 2):
                # Selection
                parents = self.selection(population, fitness)

                # Crossover
                offspring = self.crossover(parents, pc)

                # Mutation
                offspring[0] = self.mutation(
                    offspring[0], pm, problem.x_lower, problem.x_upper
                )
                offspring[1] = self.mutation(
                    offspring[1], pm, problem.x_lower, problem.x_upper
                )

                new_population.extend(offspring)

            population = np.array(new_population)

            if nfe >= self.nfe_max:
                break

        return best_solution, best_fitness
