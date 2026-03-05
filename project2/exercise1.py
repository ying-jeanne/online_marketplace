import numpy as np
import matplotlib.pyplot as plt
import os


def epsilon_greedy(T, epsilon_schedule, true_probs, rng):
    """
    Run epsilon-greedy algorithm for T steps.

    Args:
        T: number of time steps
        epsilon_schedule: either a float (fixed) or 'decay' for 1/sqrt(t)
        true_probs: list of true Bernoulli reward probabilities for each arm
        rng: numpy random generator

    Returns:
        actions: array of chosen actions at each step
        rewards: array of observed rewards at each step
    """
    K = len(true_probs)
    # Estimated reward for each arm
    Q = np.zeros(K)
    # Number of times each arm has been pulled
    N_pulls = np.zeros(K)

    actions = np.zeros(T, dtype=int)
    rewards = np.zeros(T)

    # We pull the arm at least once to initialize the expected reward and N_pulls
    for a in range(K):
        # We assume T >= K for this to make sense
        if a < T:
            action = a
            reward = 1.0 if rng.random() < true_probs[action] else 0.0
            
            N_pulls[action] = 1
            Q[action] = reward
            
            actions[a] = action
            rewards[a] = reward

    # We loop through T pulls, update expected reward and N_pulls
    for t in range(K, T):
        # Determine epsilon for this step (note: t is 0-indexed, so t+1 is the true time step)
        if epsilon_schedule == "decay":
            eps = 1.0 / np.sqrt(t + 1)
        else:
            eps = epsilon_schedule

        # Epsilon-greedy action selection
        if rng.random() < eps:
            action = rng.integers(0, K)
        else:
            max_q = np.max(Q)
            best_arms = np.where(Q == max_q)[0]
            action = rng.choice(best_arms)

        # Play and observe
        reward = 1.0 if rng.random() < true_probs[action] else 0.0

        # Update estimates (using exact formula from lecture slide)
        # ˆμat = (ˆμat * Nat + rat,t) / (Nat + 1)
        Q[action] = (Q[action] * N_pulls[action] + reward) / (N_pulls[action] + 1)
        N_pulls[action] += 1

        actions[t] = action
        rewards[t] = reward

    return actions, rewards

def run_simulations(T, epsilon_schedule, true_probs, n_sims=500, seed=42):
    """
    Run multiple simulations of epsilon-greedy.

    Returns:
        best_arm_fraction: fraction of times best arm is selected at each t
        avg_regret: average cumulative regret at each t
    """
    best_arm = np.argmax(true_probs)
    best_prob = true_probs[best_arm]

    best_arm_counts = np.zeros(T)
    cumulative_regrets = np.zeros(T)

    for sim in range(n_sims):
        rng = np.random.default_rng(seed + sim)
        actions, rewards = epsilon_greedy(T, epsilon_schedule, true_probs, rng)

        # Track best arm selection
        best_arm_selected = (actions == best_arm).astype(float)
        best_arm_counts += best_arm_selected

        # Compute cumulative regret: sum of (best_prob - prob of chosen arm)
        per_step_regret = np.array([best_prob - true_probs[a] for a in actions])
        cumulative_regrets += np.cumsum(per_step_regret)

    best_arm_fraction = best_arm_counts / n_sims
    avg_regret = cumulative_regrets / n_sims

    return best_arm_fraction, avg_regret


def compute_interval_percentages(best_arm_fraction, interval=100):
    """
    Compute average percentage of best arm selection in intervals.
    E.g., intervals of 100: [0-100), [100-200), ..., [900-1000)
    """
    T = len(best_arm_fraction)
    n_intervals = T // interval
    percentages = []

    for i in range(n_intervals):
        start = i * interval
        end = (i + 1) * interval
        pct = best_arm_fraction[start:end].mean() * 100
        percentages.append(pct)

    return percentages


def plot_best_arm_intervals(results, T, interval=100, display=False):
    """Plot percentage of best arm selected in intervals of 100 steps."""
    os.makedirs("figures", exist_ok=True)

    n_intervals = T // interval
    x_labels = [f"{i*interval}-{(i+1)*interval}" for i in range(n_intervals)]
    x_pos = np.arange(n_intervals)
    bar_width = 0.25

    plt.figure(figsize=(12, 6))

    for idx, (label, (fraction, _)) in enumerate(results.items()):
        percentages = compute_interval_percentages(fraction, interval)
        plt.bar(x_pos + idx * bar_width, percentages, bar_width,
                label=label, alpha=0.8)

    plt.xlabel("Time Interval")
    plt.ylabel("% Best Arm Selected")
    plt.title("Percentage of Best Arm Selected per 100-Step Interval")
    plt.xticks(x_pos + bar_width, x_labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if display:
        plt.show()
    else:
        plt.savefig("figures/best_arm_fraction.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_regret(results_subset, T, display=False):
    """Plot average cumulative regret over time for fixed epsilon only."""
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, (_, regret) in results_subset.items():
        plt.plot(range(1, T + 1), regret, label=label)

    plt.xlabel("Time Step (T)")
    plt.ylabel("Average Cumulative Regret")
    plt.title("Average Cumulative Regret Over Time (500 Simulations)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if display:
        plt.show()
    else:
        plt.savefig("figures/regret_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    # Problem parameters
    true_probs = [0.3, 0.4, 0.5]  # Bernoulli reward probabilities
    T = 1000
    n_sims = 500

    # Three epsilon schedules
    schedules = {
        "eps = 0.1 (fixed)": 0.1,
        "eps = 0.2 (fixed)": 0.2,
        "eps = 1/sqrt(t) (decay)": "decay",
    }

    results = {}
    for label, schedule in schedules.items():
        print(f"Running {label}...")
        fraction, regret = run_simulations(T, schedule, true_probs, n_sims)
        results[label] = (fraction, regret)

        # Report summary per 100-step intervals
        percentages = compute_interval_percentages(fraction)
        print(f"  Best arm % per interval: {[f'{p:.1f}%' for p in percentages]}")
        print(f"  Final cumulative regret: {regret[-1]:.2f}")

    # Plot 1: Best arm fraction in intervals (all 3 schedules)
    plot_best_arm_intervals(results, T)

    # Plot 2: Regret (only fixed epsilon = 0.1 and 0.2)
    regret_subset = {k: v for k, v in results.items() if "fixed" in k}
    plot_regret(regret_subset, T)

    print("\nFigures saved to figures/")


if __name__ == "__main__":
    main()
