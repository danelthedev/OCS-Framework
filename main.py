import numpy as np
from random_search.random_searcher import RandomSearcher
from functions.shifted_elliptic import ShiftedElliptic
from functions.shifted_sphere import ShiftedSphere
from functions.shifted_schwefel import ShiftedSchwefel

from genetic_algorithms.cga import CGA


def random_search_test(
    function_name, optimizer, dimension=10, max_iter=100, population_size=30, alfa=5
):
    print("-" * 50)
    print(f"\nTesting {function_name.__name__}")
    print(f"Using optimizer {optimizer}")

    func = lambda x: function_name.func(x)

    # Initialize the random searcher with updated bounds [-100, 100]
    searcher = RandomSearcher(
        max_iter=max_iter,
        alfa=alfa,
        func=func,
        population_size=population_size,
        dimension=dimension,
    )

    # Run optimization
    if optimizer == "optimize_Population_V1_selfAdaptive":
        best_solution, best_fitness = searcher.optimize_Population_V1_selfAdaptive()
    elif optimizer == "optimize_Population_V3":
        best_solution, best_fitness = searcher.optimize_Population_V3()

    print(f"\nFinal Results:")
    print(f"Best fitness: {best_fitness}")
    print(f"Best solution: {best_solution}")
    print("-" * 50)


def genetic_test():

    # Initialize and run the genetic algorithm
    ga = CGA(fitness_function=ShiftedSchwefel.func, nfe=10000)
    best_solution, best_fitness = ga.run()

    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)


def main():
    genetic_test()


if __name__ == "__main__":
    main()
