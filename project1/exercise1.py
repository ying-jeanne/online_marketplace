import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os

def generate_positions(n, rng):
    return rng.random((n, 2))

def generate_sample_data(n, rng):
    """Generate a single sample of positions and distance matrix"""
    riders = generate_positions(n, rng)
    drivers = generate_positions(n, rng)
    dist = cdist(riders, drivers, metric="euclidean")
    return riders, drivers, dist

def random_matching(dist, rng):
    n = dist.shape[0]
    perm = rng.permutation(n)
    return dist[np.arange(n), perm].sum()

def greedy_matching(dist):
    n = dist.shape[0]
    remaining_rows = set(range(n))
    remaining_cols = set(range(n))
    total = 0.0

    while remaining_rows:
        best = (None, None, float("inf"))
        for i in remaining_rows:
            for j in remaining_cols:
                d = dist[i, j]
                if d < best[2]:
                    best = (i, j, d)

        i, j, d = best
        total += d
        remaining_rows.remove(i)
        remaining_cols.remove(j)

    return total

def optimal_matching(dist):
    r, c = linear_sum_assignment(dist)  #Using the Hungarian algorithm
    return dist[r, c].sum()

def run_simulations(n, sims=1000, seed=1):
    rng = np.random.default_rng(seed)
    out = {"random": [], "greedy": [], "optimal": []}

    for _ in range(sims):
        _, _, dist = generate_sample_data(n, rng)
        out["random"].append(random_matching(dist, rng))
        out["greedy"].append(greedy_matching(dist))
        out["optimal"].append(optimal_matching(dist))

    # converting to arrays
    return {k: np.array(v) for k, v in out.items()}

def plot_distributions(results_by_n, display=False):
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    for n, res in results_by_n.items():
        plt.figure()
        for method in ["random", "greedy", "optimal"]:
            plt.hist(res[method], bins=30, density=True, alpha=0.6, label=method)
        plt.title(f"Total distance distribution (n={n})")
        plt.xlabel("Total distance")
        plt.ylabel("Density")
        plt.legend()
        if display:
            plt.show()
        else:
            plt.savefig(os.path.join("figures", f"total_distance_distribution_n_{n}.png"), dpi=300, bbox_inches='tight')
            plt.close()

def main():
    # Generate sample distance matrix for n=10
    print("=== Sample Distance Matrix (n=10) ===")
    rng = np.random.default_rng(1)
    _, _, dist = generate_sample_data(10, rng)
    print(f"Sample distance matrix (first 5x5):\n{dist[:5, :5]}\n")

    # Run simulations
    results = {
        10: run_simulations(10, sims=1000, seed=1),
        30: run_simulations(30, sims=1000, seed=2),
        50: run_simulations(50, sims=1000, seed=3),
    }

    # Printing averages and graph
    for n in [10, 30, 50]:
        print(f"\n=== n={n} ===")
        for method in ["random", "greedy", "optimal"]:
            print(f"{method:>7}: mean={results[n][method].mean():.4f}")

    plot_distributions(results)

if __name__ == "__main__":
    main()
