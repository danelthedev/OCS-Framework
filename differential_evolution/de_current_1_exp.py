import numpy as np


def differential_evolution_current_1_exp(func, bounds, F, CR, pop_size, max_nfe):
    """
    Current-to-1 Exponential DE variant (DE/current/1/exp)

    Args:
        func: Objective function to minimize
        bounds: Array of bounds for each dimension [(min, max), ...]
        F: Mutation scale factor (typically 0.5-1.0)
        CR: Crossover probability (typically 0.7-1.0)
        pop_size: Population size
        max_nfe: Maximum number of function evaluations
    """
    # Get problem dimension from bounds
    dim = len(bounds)

    # Initialize population randomly within bounds
    pop = np.random.rand(pop_size, dim)
    for i in range(dim):
        pop[:, i] = bounds[i][0] + pop[:, i] * (bounds[i][1] - bounds[i][0])

    nfe = 0  # Function evaluation counter
    while nfe < max_nfe:
        for i in range(pop_size):
            # Select 3 random distinct vectors for mutation, excluding current vector
            indices = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]

            # Create mutant vector using DE/current/1 strategy
            # mutant = a + F * (b - c)
            mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])

            # Exponential crossover
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one dimension is crossed over
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            # Create trial vector
            trial = np.where(cross_points, mutant, pop[i])

            # Selection: keep trial if it's better
            f_trial = func(trial)
            f_target = func(pop[i])
            nfe += 1

            if f_trial < f_target:  # Minimization
                pop[i] = trial

    # Return best solution found
    best_idx = np.argmin([func(ind) for ind in pop])
    return pop[best_idx]
