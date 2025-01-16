import numpy as np


def control_x(x, x_lower, x_upper):
    x = np.clip(x, x_lower, x_upper)
    return x


def f1(x, o):
    z = np.array(x) - np.array(o)
    sum_z_sqared = np.sum(z**2)
    return sum_z_sqared


def initialize_population(pop_size, dim, x_lower, x_upper):
    return np.random.uniform(x_lower, x_upper, (pop_size, dim))


def modify_x(x, alpha, dim, x_lower, x_upper):
    new_agents = []
    for _ in range(dim):
        rand_vector = np.random.uniform(-1, 1, dim)
        x_m = x + alpha * rand_vector
        x_m = control_x(x_m, x_lower, x_upper)
        new_agents.append(x_m)
    return np.array(new_agents)


def optimizer(
    pop_size, max_pop_size, dim, x_lower, x_upper, max_iter, o, alpha_init=1.0
):
    population = initialize_population(pop_size, dim, x_lower, x_upper)
    convergence_history = []
    best_fitness_ever = float("inf")
    total_evals = 0
    max_evals = pop_size * (dim + 1) * max_iter  # Total allowed evaluations

    for iteration in range(max_iter):
        if total_evals >= max_evals:
            break

        fitness = np.array([f1(ind, o) for ind in population])
        total_evals += len(population)

        current_best = np.min(fitness)
        best_fitness_ever = min(best_fitness_ever, current_best)
        convergence_history.append(best_fitness_ever)

        alpha = alpha_init / (iteration + 1)
        new_population = []

        for i in range(len(population)):
            if total_evals >= max_evals:
                break

            agents = modify_x(population[i], alpha, dim, x_lower, x_upper)
            agent_fitness = np.array([f1(agent, o) for agent in agents])
            total_evals += len(agents)

            # Keep only the best offspring if better than parent
            better_agents = agents[agent_fitness < fitness[i]]
            if len(better_agents) > 0:
                best_idx = np.argmin(agent_fitness[agent_fitness < fitness[i]])
                new_population.append(better_agents[best_idx])
            else:
                new_population.append(population[i])

        population = np.array(new_population)

        if iteration % 100 == 0:
            print(
                f"Iteration {iteration + 1}, Population size: {len(population)}, Best Fitness: {best_fitness_ever}, Evals: {total_evals}"
            )

    best_index = np.argmin([f1(ind, o) for ind in population])
    return population[best_index], f1(population[best_index], o), convergence_history


d = 10
x_lower = -100
x_upper = 100
max_iter = 1_000_100
pop_size = 50
max_pop_size = 1_000
o = np.random.uniform(x_lower, x_upper, d)
best_solution, best_fitness, convergence_history = optimizer(
    pop_size, max_pop_size, d, x_lower, x_upper, max_iter, o
)


print(best_solution, best_fitness)
