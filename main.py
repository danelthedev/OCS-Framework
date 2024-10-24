import numpy as np
from random_search.random_searcher import RandomSearcher
from functions.shifted_elliptic import ShiftedElliptic
from functions.shifted_sphere import ShiftedSphere


def test_function(
    function_name, dimension=10, max_iter=100, population_size=30, alfa=5
):
    print(f"\nTesting {function_name.__name__}")
    print("-" * 50)

    # Create random shift vector and rotation matrix for shifted elliptic
    if function_name == ShiftedElliptic:
        o = np.zeros(dimension)  # shift vector should be 0 (global optimum at 0)
        M = np.random.randn(dimension, dimension)  # rotation matrix
        func = lambda x: ShiftedElliptic.func(x, o, M)

    # Initialize the random searcher with updated bounds [-100, 100]
    searcher = RandomSearcher(
        max_iter=max_iter,
        alfa=alfa,
        func=func,
        population_size=population_size,
        dimension=dimension,
    )

    # Run optimization
    best_solution, best_fitness = searcher.optimize_Population_V3()

    print(f"\nFinal Results:")
    print(f"Best fitness: {best_fitness}")
    print(f"Best solution: {best_solution}")


def main():
    # Test parameters
    dimension = 5
    max_iter = 100
    population_size = 30
    alfa = 5  # number of offspring per parent

    # Test both functions
    test_function(ShiftedElliptic, dimension, max_iter, population_size, alfa)


if __name__ == "__main__":
    main()
