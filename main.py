import numpy as np

from random_search.random_searcher import RandomSearcher
from functions.shifted_schwefel import ShiftedSchwefel

from genetic_algorithms.cga import CGA

from differential_evolution.de_current_1_exp import differential_evolution_current_1_exp
from differential_evolution.de_rand_1_bin import differential_evolution_rand1_bin

from genetic_algorithms.cga_adaptivev1 import CGAAdaptiveV1
from functions.shifted_elliptic import ShiftedElliptic
from real_coded_ga.rga_3 import RGA_3


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


def differential_evolution_test():
    # Example usage
    def sphere(x):
        return sum(x ** 2)

    bounds = [(-100, 100)] * 20
    best_solution = differential_evolution_rand1_bin(sphere, np.array(bounds), F=0.8, CR=0.9, pop_size=20, max_nfe=1000)
    print("Best solution:", best_solution, "Fitness:", sphere(best_solution))

    best_solution = differential_evolution_rand1_bin(sphere, np.array(bounds), F=0.8, CR=0.9, pop_size=20, max_nfe=1000)
    print("Best solution:", best_solution, "Fitness:", sphere(best_solution))


def cga_adaptive_test():
    # Initialize problem (2 dimensions)
    problem = ShiftedElliptic(x_lower=[-100] * 2, x_upper=[100] * 2)

    cga = CGAAdaptiveV1(pop_size=50, initial_pc=0.8, initial_pm=0.1, nfe_max=1000)
    best_solution, best_fitness = cga.optimize(problem)

    print("CGA Adaptive V1 Results:")
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)


def rga3_test():
    # Initialize problem (2 dimensions)
    problem = ShiftedElliptic(x_lower=[-100] * 3, x_upper=[100] * 3)

    bounds = list(zip(problem.x_lower, problem.x_upper))

    rga = RGA_3(bounds=bounds, pop_size=50, pc=0.8, pm=0.1, nfe=1000)
    best_solution, best_fitness = rga.optimize(problem.func)

    print("\nRGA_3 Results:")
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)


def rga1_adaptive_test():
    # Initialize problem (2 dimensions)
    problem = ShiftedElliptic(x_lower=[-100] * 2, x_upper=[100] * 2)

    bounds = list(zip(problem.x_lower, problem.x_upper))

    rga = RGA_3(bounds=bounds, pop_size=50, pc=0.8, pm=0.1, nfe=1000)
    best_solution, best_fitness = rga.optimize(problem.func)

    print("\nRGA_1 Adaptive Results:")
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)


def main():
    # differential_evolution_test()
    # cga_adaptive_test()
    # rga3_test()
    rga1_adaptive_test()

if __name__ == "__main__":
    main()
