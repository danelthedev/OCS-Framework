import numpy as np


def control_x(x, x_lower, x_upper):
    x=np.clip(x,x_lower,x_upper)
    return x


def f1(x,o):
    z = np.array(x) - np.array(o)
    sum_z_sqared = np.sum(z**2)
    return sum_z_sqared


def initialize_population(pop_size, dim, x_lower, x_upper):
    return np.random.uniform(x_lower,x_upper,(pop_size,dim))


def modify_x(x,alpha,dim,x_lower,x_upper):
    new_agents = []
    for _ in range(dim):
        rand_vector = np.random.uniform(-1,-1,dim)
        x_m = x + alpha * rand_vector
        x_m = control_x(x_m,x_lower,x_upper)
        new_agents.append(x_m)
    return np.array(new_agents)


def optimizer(pop_size,max_pop_size,dim,x_lower,x_upper,max_iter,o,alpha_init=1.0):
    population = initialize_population(pop_size,dim,x_lower,x_upper)
    alpha = alpha_init
    for iteration in range(max_iter):
        fitness = np.array([f1(ind,o) for ind in population])
        alpha = alpha_init / (iteration + 1)
        new_population = []
        for i in range(pop_size):
            agents = modify_x(population[i],alpha,dim,x_lower,x_upper)
            agent_fitness = np.array([f1(agent,o) for agent in agents])
            better_agents = agents[agent_fitness < fitness[i]]
            if len(better_agents) > 0:
                new_population.extend(better_agents)
            else:
                new_population.append(population[i])
        population = np.array(new_population)
        pop_size = len(population)
        if pop_size > max_pop_size:
            sorted_indices = np.argsort([f1(ind,o) for ind in population])
            population = population[sorted_indices[:max_pop_size]]
            pop_size = max_pop_size
        if len(population) == 0:
            print(f'extint at population {iteration}')
            break
        print(f'Iteration {iteration + 1}, Population size: {pop_size}, Best Fitness: {np.min(fitness)}')
    best_index = np.argmin(fitness)
    return population[best_index],fitness[best_index]



d = 10
x_lower = -100
x_upper = 100
max_iter = 1_000_100
pop_size = 50
max_pop_size = 1_000
o = np.random.uniform(x_lower,x_upper,d)
best_solution, best_fitness = optimizer(pop_size, max_pop_size, d, x_lower, x_upper, max_iter, o)


print(best_solution, best_fitness)



