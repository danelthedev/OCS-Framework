import numpy as np


def differential_evolution_rand1_bin(func, bounds, F, CR, pop_size, max_nfe):
    dim = len(bounds)
    pop = np.random.rand(pop_size, dim)
    for i in range(dim):
        pop[:, i] = bounds[i][0] + pop[:, i] * (bounds[i][1] - bounds[i][0])

    nfe = 0
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

    best_idx = np.argmin([func(ind) for ind in pop])
    return pop[best_idx]