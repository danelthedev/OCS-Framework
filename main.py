import numpy as np
from random_search.random_searcher import RandomSearcher
from functions.shifted_elliptic import ShiftedElliptic
from functions.shifted_sphere import ShiftedSphere


def test_function(
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


def main():
    # Test parameters
    dimension = 5
    max_iter = 100
    population_size = 30
    alfa = 5  # number of offspring per parent

    # Test both functions
    test_function(ShiftedElliptic, "optimize_Population_V1_selfAdaptive", dimension, max_iter, population_size, alfa)
    test_function(ShiftedElliptic, "optimize_Population_V3", dimension, max_iter, population_size, alfa)


if __name__ == "__main__":
    main()
