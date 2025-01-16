import numpy as np
from typing import Tuple, List


def differential_evolution_rand1_bin(
    func, bounds, F=0.8, CR=0.9, pop_size=50, max_nfe=1000
) -> Tuple[np.ndarray, List[float]]:
    dim = len(bounds)
    pop = np.random.rand(pop_size, dim)
    for i in range(dim):
        pop[:, i] = bounds[i][0] + pop[:, i] * (bounds[i][1] - bounds[i][0])

    # Initialize convergence history
    convergence_history = []
    best_fitness_so_far = float("inf")

    # Initial population evaluation
    fitness = np.array([func(ind) for ind in pop])
    best_fitness_so_far = min(best_fitness_so_far, np.min(fitness))
    convergence_history.append(best_fitness_so_far)

    nfe = pop_size
    while nfe < max_nfe:
        for i in range(pop_size):
            indices = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])

            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            trial = np.where(cross_points, mutant, pop[i])
            f_trial = func(trial)
            f_target = func(pop[i])
            nfe += 1

            if f_trial < f_target:
                pop[i] = trial
                best_fitness_so_far = min(best_fitness_so_far, f_trial)
            else:
                best_fitness_so_far = min(best_fitness_so_far, f_target)

            convergence_history.append(best_fitness_so_far)

            if nfe >= max_nfe:
                break

    best_idx = np.argmin([func(ind) for ind in pop])
    return pop[best_idx], convergence_history
