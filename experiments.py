import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Remade.L2 import PopulationV3Adaptive
from random_search.random_searcher import RandomSearcher
from functions.shifted_elliptic import ShiftedElliptic
from functions.shifted_schwefel import ShiftedSchwefel
from Remade.L3 import CGAAdaptiveV1Replacement
from Remade.L4 import RGA4Adaptive
import seaborn as sns
from genetic_algorithms.cga import CGA
from genetic_algorithms.cga_adaptivev1 import CGAAdaptiveV1
from real_coded_ga.rga_3 import RGA_3
from real_coded_ga.rga_1_adaptive import RGA_1_Adaptive
from Remade.L5 import DEBest2Exp
from differential_evolution.de_rand_1_bin import differential_evolution_rand1_bin
from differential_evolution.de_current_1_exp import differential_evolution_current_1_exp
from tabulate import tabulate
from functions.shifted_sphere import ShiftedSphere
from functions.noisy_schwefel import NoisySchwefel
from functions.schwefel_bounds import SchwefelBounds
from scipy import stats as scipy_stats


def run_experiment(
    algorithm_class, D, n_runs=10, nef=1000, run_method="run", **algo_params
):
    results = []
    convergence_data = []

    population_size = algo_params.get("population_size", 10)

    # Handle both max_iter and max_nfe
    if "max_nfe" in algo_params:
        algo_params["max_nfe"] = nef  # Override max_nfe with NEF
    else:
        algo_params["max_iter"] = nef // population_size

    # Create ShiftedElliptic function instance
    obj_function = ShiftedElliptic(x_lower=[-100] * D, x_upper=[100] * D)

    for run in range(n_runs):
        optimizer = algorithm_class(
            obj_function=obj_function.evaluate,  # Use evaluate method from BaseFunction
            lower_bounds=[-100] * D,
            upper_bounds=[100] * D,
            **algo_params,
        )

        # Use the specified run method
        run_func = getattr(optimizer, run_method)
        best_solution, best_fitness = run_func()
        results.append(best_fitness)

        if run == 0:
            convergence_data = optimizer.convergence_history

    stats = pd.Series(
        {
            "min": np.min(results),
            "max": np.max(results),
            "mean": np.mean(results),
            "std": np.std(results),
        }
    )

    return stats, convergence_data


def plot_convergence(convergence_data, algorithm_name, D):
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_data, label=f"{algorithm_name} (D={D})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.yscale("log")
    plt.title(f"Convergence Plot for {algorithm_name}")
    plt.legend()
    plt.grid(True)
    return plt


# Modify CGAAdaptiveV1 wrapper to match our interface
class CGAAdaptiveV1Wrapper(CGAAdaptiveV1):
    def __init__(
        self,
        obj_function,
        lower_bounds,
        upper_bounds,
        population_size,
        max_nfe,
        pc_initial=0.8,
        pm_initial=0.1,
    ):
        # Store initial values before super().__init__
        self.initial_pc = pc_initial
        self.initial_pm = pm_initial

        super().__init__(
            pop_size=population_size,
            initial_pc=pc_initial,
            initial_pm=pm_initial,
            nfe_max=max_nfe,
        )
        self.obj_function = obj_function
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def run(self):
        # Create a problem-like object that CGAAdaptiveV1 expects
        class Problem:
            def __init__(self, f, x_lower, x_upper):
                self.f = f
                self.x_lower = x_lower
                self.x_upper = x_upper

        problem = Problem(self.obj_function, self.lower_bounds, self.upper_bounds)
        best_solution, best_fitness = self.optimize(problem)

        # Normalize convergence history to 100 points if needed
        total_points = len(self.convergence_history)
        if total_points != 100:
            indices = np.linspace(0, total_points - 1, 100, dtype=int)
            self.convergence_history = [self.convergence_history[i] for i in indices]

        return best_solution, best_fitness


class DECurrent1ExpWrapper:
    def __init__(
        self,
        obj_function,
        lower_bounds,
        upper_bounds,
        population_size,
        max_nfe,
        F=0.5,
        CR=0.7,
    ):
        self.obj_function = obj_function
        self.bounds = np.array(list(zip(lower_bounds, upper_bounds)))
        self.F = F
        self.CR = CR
        self.pop_size = population_size
        self.max_nfe = max_nfe
        self.convergence_history = []

    def run(self):
        best_solution, convergence = differential_evolution_current_1_exp(
            func=self.obj_function,
            bounds=self.bounds,
            F=self.F,
            CR=self.CR,
            pop_size=self.pop_size,
            max_nfe=self.max_nfe,
        )

        # Normalize convergence history to 100 points
        total_points = len(convergence)
        indices = np.linspace(0, total_points - 1, 100, dtype=int)
        self.convergence_history = [convergence[i] for i in indices]

        return best_solution, self.obj_function(best_solution)


class RGA3Wrapper:
    def __init__(
        self,
        obj_function,
        lower_bounds,
        upper_bounds,
        population_size,
        max_nfe,
        pc=0.8,
        pm=0.1,
    ):
        self.bounds = list(zip(lower_bounds, upper_bounds))
        self.optimizer = RGA_3(
            bounds=self.bounds, pop_size=population_size, pc=pc, pm=pm, nfe=max_nfe
        )
        self.obj_function = obj_function
        self.convergence_history = []

    def run(self):
        # Negate objective function for maximization (RGA_3 maximizes by default)
        neg_obj_func = lambda x: -self.obj_function(x)
        best_solution, best_fitness = self.optimizer.optimize(neg_obj_func)
        # Negate back the convergence history and best fitness
        self.convergence_history = [-x for x in self.optimizer.convergence_history]
        return best_solution, -best_fitness


class DERand1BinWrapper:
    def __init__(
        self,
        obj_function,
        lower_bounds,
        upper_bounds,
        population_size,
        max_nfe,
        F=0.8,
        CR=0.9,
    ):
        self.obj_function = obj_function
        self.bounds = np.array(list(zip(lower_bounds, upper_bounds)))
        self.F = F
        self.CR = CR
        self.pop_size = population_size
        self.max_nfe = max_nfe
        self.convergence_history = []

    def run(self):
        best_solution, convergence = differential_evolution_rand1_bin(
            func=self.obj_function,
            bounds=self.bounds,
            F=self.F,
            CR=self.CR,
            pop_size=self.pop_size,
            max_nfe=self.max_nfe,
        )

        # Normalize convergence history to 100 points
        total_points = len(convergence)
        indices = np.linspace(0, total_points - 1, 100, dtype=int)
        self.convergence_history = [convergence[i] for i in indices]

        return best_solution, self.obj_function(best_solution)


# Add wrapper class for RGA_1_Adaptive
class RGA1AdaptiveWrapper:
    def __init__(
        self, obj_function, lower_bounds, upper_bounds, population_size, max_nfe
    ):
        self.bounds = list(zip(lower_bounds, upper_bounds))
        self.optimizer = RGA_1_Adaptive(
            bounds=self.bounds,
            pop_size=population_size,
            pc_start=0.9,
            pc_end=0.6,
            pm_start=0.1,
            pm_end=0.01,
            nfe=max_nfe,
        )
        self.obj_function = obj_function
        self.convergence_history = []
        self.best_so_far = float("inf")

    def run(self):
        neg_obj_func = lambda x: -self.obj_function(x)
        best_solution, best_fitness, history = self.optimizer.optimize(
            neg_obj_func
        )  # Assuming optimize returns history

        # Store the convergence history
        self.convergence_history = [-x for x in history]

        # Normalize to 100 points
        if len(self.convergence_history) > 100:
            total_points = len(self.convergence_history)
            indices = np.linspace(0, total_points - 1, 100, dtype=int)
            self.convergence_history = [self.convergence_history[i] for i in indices]

        return best_solution, -best_fitness


# Run experiments for both functions
functions = {
    "ShiftedElliptic": ShiftedElliptic,
    "ShiftedSchwefel": ShiftedSchwefel,
    "ShiftedSphere": ShiftedSphere,
    "NoisySchwefel": NoisySchwefel,
    "SchwefelBounds": SchwefelBounds,
}

dimensions = [5, 10]
algorithms = {
    "Population_V3_adaptive": {
        "class": PopulationV3Adaptive,
        "params": {"population_size": 10, "alpha_initial": 1.0},
    },
    "Population_V1_selfAdaptive": {
        "class": lambda **kwargs: RandomSearcher(
            max_iter=kwargs["max_iter"],
            alfa=1.0,
            func=kwargs["obj_function"],
            population_size=10,
            dimension=len(kwargs["lower_bounds"]),
            print_results=False,
        ),
        "params": {},
        "run_method": "optimize_Population_V1_selfAdaptive",
    },
    "Population_V3": {
        "class": lambda **kwargs: RandomSearcher(
            max_iter=kwargs["max_iter"],
            alfa=3,
            func=kwargs["obj_function"],
            population_size=10,
            dimension=len(kwargs["lower_bounds"]),
            print_results=False,
        ),
        "params": {},
        "run_method": "optimize_Population_V3",
    },
    "CGA_adaptiveV1_replacement": {
        "class": CGAAdaptiveV1Replacement,
        "params": {
            "population_size": 20,
            "pc_initial": 0.8,
            "pm_initial": 0.1,
            "max_nfe": 1000,
        },
    },
    "CGA": {
        "class": lambda **kwargs: CGA(
            fitness_function=kwargs["obj_function"],
            lower_bounds=kwargs["lower_bounds"],
            upper_bounds=kwargs["upper_bounds"],
            pc=0.8,
            pm=0.05,
            nfe=kwargs["max_nfe"],
            population_size=20,
        ),
        "params": {"max_nfe": 1000},
    },
    "CGAAdaptiveV1_Original": {
        "class": CGAAdaptiveV1Wrapper,
        "params": {
            "population_size": 20,
            "pc_initial": 0.8,
            "pm_initial": 0.1,
            "max_nfe": 1000,
        },
    },
    "RGA_4_adaptive": {
        "class": RGA4Adaptive,
        "params": {
            "population_size": 20,
            "pc_initial": 0.8,
            "pm_initial": 0.1,
            "max_nfe": 1000,
        },
    },
    "RGA_3": {
        "class": RGA3Wrapper,
        "params": {"population_size": 20, "pc": 0.8, "pm": 0.1, "max_nfe": 1000},
    },
    "RGA_1_adaptive": {
        "class": RGA1AdaptiveWrapper,
        "params": {"population_size": 20, "max_nfe": 1000},
    },
    "DE/best/2/exp": {
        "class": DEBest2Exp,
        "params": {
            "population_size": 20,
            "F": 0.8,  # Scale factor
            "CR": 0.9,  # Crossover rate
            "max_nfe": 1000,
        },
    },
    "DE/rand/1/bin": {
        "class": DERand1BinWrapper,
        "params": {"population_size": 20, "F": 0.8, "CR": 0.9, "max_nfe": 1000},
    },
    "DE/current/1/exp": {
        "class": lambda **kwargs: DECurrent1ExpWrapper(
            obj_function=kwargs["obj_function"],
            lower_bounds=kwargs["lower_bounds"],
            upper_bounds=kwargs["upper_bounds"],
            population_size=kwargs["population_size"],
            max_nfe=kwargs["max_nfe"],
            F=kwargs["F"],
            CR=kwargs["CR"],
        ),
        "params": {
            "population_size": 20,
            "F": 0.5,  # Different from other DE variants
            "CR": 0.7,  # Different from other DE variants
            "max_nfe": 1000,
        },
    },
}

# Run experiments and collect results
all_results = {}
all_convergence_plots = {}

for func_name, func_class in functions.items():
    results = {}
    convergence_plots = {}

    for D in dimensions:
        obj_function = func_class(x_lower=[-100] * D, x_upper=[100] * D)

        for algo_name, algo_info in algorithms.items():
            stats, convergence = run_experiment(
                algo_info["class"],
                D,
                n_runs=10,
                nef=1000,
                run_method=algo_info.get("run_method", "run"),
                **algo_info["params"],
            )

            key = f"{algo_name}_D{D}"
            results[key] = stats
            convergence_plots[key] = convergence

    all_results[func_name] = pd.DataFrame(results)
    all_convergence_plots[func_name] = convergence_plots

# Define distinct color palette and line styles
colors = sns.color_palette("husl", n_colors=len(algorithms))  # Highly distinct colors
line_styles = ["-", "--", ":", "-."] * (
    len(algorithms) // 4 + 1
)  # Different line styles

# Plot separate graphs for each function
for func_name, convergence_plots in all_convergence_plots.items():
    # Create two subplots side by side (one for each dimension)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot D=5 on first subplot
    for i, algo_name in enumerate(algorithms.keys()):
        key = f"{algo_name}_D5"
        ax1.plot(
            convergence_plots[key],
            label=algo_name,
            color=colors[i],
            linestyle=line_styles[i],
            linewidth=2,
        )

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best Fitness")
    ax1.set_yscale("log")
    ax1.set_title(f"{func_name} (D=5)")
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Plot D=10 on second subplot
    for i, algo_name in enumerate(algorithms.keys()):
        key = f"{algo_name}_D10"
        ax2.plot(
            convergence_plots[key],
            label=algo_name,
            color=colors[i],
            linestyle=line_styles[i],
            linewidth=2,
        )

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Fitness")
    ax2.set_yscale("log")
    ax2.set_title(f"{func_name} (D=10)")
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Add a single legend for both plots
    lines = ax2.get_lines()
    labels = [line.get_label() for line in lines]
    fig.legend(
        lines,
        labels,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        title="Algorithms",
        borderaxespad=0.5,
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust layout to make room for legend

    # Save the figure
    plt.savefig(f"convergence_plot_{func_name}.png", dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    # Print numerical results in a table
    print(f"\nNumerical Results for {func_name}:")
    print(all_results[func_name].round(6))

# After running experiments, create formatted tables
for func_name, results_df in all_results.items():
    print(f"\n{func_name} Performance Metrics")

    for D in dimensions:
        table_data = []
        for algo_name in algorithms.keys():
            key = f"{algo_name}_D{D}"
            stats = results_df[key]
            table_data.append(
                [
                    algo_name,
                    f"{stats['min']:.2e}",
                    f"{stats['max']:.2e}",
                    f"{stats['mean']:.2e}",
                    f"{stats['std']:.2e}",
                ]
            )

        print(f"\nDimension: {D}")
        print(
            tabulate(
                table_data,
                headers=["Algorithm", "Minimum", "Maximum", "Mean", "Std Dev"],
                tablefmt="grid",
                stralign="left",
            )
        )


def create_table_plot(data, headers, title, filename):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, len(data) * 0.5 + 1))

    # Hide axes
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#f2f2f2"] * len(headers),
        cellColours=[["#ffffff"] * len(headers)] * len(data),
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Add title
    plt.title(title, pad=20)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close()


def format_number(num):
    """Format number to avoid scientific notation and limit decimal places"""
    if abs(num) < 0.01:
        # For very small numbers, show more decimal places
        return f"{num:.6f}"
    elif abs(num) < 1:
        return f"{num:.4f}"
    elif abs(num) < 1000:
        return f"{num:.2f}"
    else:
        # For large numbers, use comma separator
        return f"{num:,.2f}"


# After running experiments, create table images
for func_name, results_df in all_results.items():
    for D in dimensions:
        table_data = []
        for algo_name in algorithms.keys():
            key = f"{algo_name}_D{D}"
            stats = results_df[key]
            table_data.append(
                [
                    algo_name,
                    format_number(stats["min"]),
                    format_number(stats["max"]),
                    format_number(stats["mean"]),
                    format_number(stats["std"]),
                ]
            )

        headers = ["Algorithm", "Minimum", "Maximum", "Mean", "Std Dev"]
        title = f"{func_name} Performance Metrics (D={D})"
        filename = f"{func_name}_D{D}_metrics"

        create_table_plot(table_data, headers, title, filename)


def compute_and_plot_ranks(data: pd.DataFrame, measure: str) -> float:
    # Calculate average ranks (ascending=True because lower values are better)
    ranks = data.rank(axis=0, ascending=True)
    avg_ranks = ranks.mean(axis=1)

    # Convert object columns to numeric arrays
    values = [data[col].apply(lambda x: float(x.item())).values for col in data.columns]
    friedman_stat, p_value = scipy_stats.friedmanchisquare(*values)

    # Create bar plot
    plt.figure(figsize=(12, 6))
    avg_ranks.plot(kind="bar")
    plt.title(f"Average Ranks of Algorithms ({measure})")
    plt.xlabel("Algorithms")
    plt.ylabel("Average Rank")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(f"average_ranks_{measure}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save ranks and stats
    avg_ranks.to_frame("Rank").to_csv(f"average_ranks_{measure}.csv")
    with open(f"statistical_analysis_{measure}.txt", "w") as f:
        f.write(f"Statistical Analysis for {measure}:\n")
        f.write(f"Friedman test statistic: {friedman_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n\n")
        f.write("Average ranks:\n")
        f.write(avg_ranks.sort_values().to_string())

    return avg_ranks


# Add after your existing results collection:
for D in dimensions:
    # Collect results for each metric
    metrics = ["min", "mean", "max", "std"]
    for metric in metrics:
        # Create DataFrame with results for current dimension and metric
        metric_results = pd.DataFrame(index=algorithms.keys(), columns=functions.keys())

        for func_name in functions:
            for algo_name in algorithms:
                key = f"{algo_name}_D{D}"
                metric_results.loc[algo_name, func_name] = all_results[func_name].loc[
                    metric, key
                ]

        # Compute and plot ranks
        avg_ranks = compute_and_plot_ranks(metric_results, f"{metric}_D{D}")

        # Perform Friedman test
        friedman_stat, p_value = scipy_stats.friedmanchisquare(
            *[metric_results[col].values for col in metric_results.columns]
        )

        print(f"\nStatistical Analysis for {metric} (D={D}):")
        print(f"Friedman test statistic: {friedman_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print("\nAverage ranks:")
        print(avg_ranks.sort_values().to_string())
