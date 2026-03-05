import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def generate_dashers(n_dashers=1000, seed=42):
    """
    Generate dasher features and true acceptance probabilities.
    
    Each dasher has features Xi = (xi1, xi2, xi3) ~ N(0,1)
    Acceptance probability: Pi = 1 / (1 + exp(-(b1*xi1 + b2*xi2 + b3*xi3)))
    b1 = 1.2, b2 = -0.8, b3 = 0.5
    """
    rng = np.random.default_rng(seed)
    
    # Feature matrix: n_dashers x 3
    X = rng.standard_normal((n_dashers, 3))
    
    # Coefficients
    beta = np.array([1.2, -0.8, 0.5])
    
    # Logit probabilities
    logits = X @ beta
    true_probs = 1.0 / (1.0 + np.exp(-logits))
    
    return X, true_probs


def random_messaging(true_probs, n_select=200, rng=None):
    """
    Random messaging: select 200 dashers at random.
    Returns selected indices and their binary responses.
    """
    n_dashers = len(true_probs)
    selected = rng.choice(n_dashers, size=n_select, replace=False)
    responses = rng.random(n_select) < true_probs[selected]
    return selected, responses.astype(int)


def thompson_sampling(alpha_params, beta_params, true_probs, n_select=200, rng=None):
    """
    Thompson Sampling: sample from Beta posteriors, select top-200.
    Returns selected indices and their binary responses.
    """
    n_dashers = len(true_probs)
    
    # Sample from Beta(alpha, beta) for each dasher
    sampled_probs = rng.beta(alpha_params, beta_params)
    
    # Select top-200 dashers with highest sampled probabilities
    selected = np.argsort(sampled_probs)[-n_select:]
    
    # Generate actual responses based on true probabilities
    responses = rng.random(n_select) < true_probs[selected]
    
    return selected, responses.astype(int)


def run_switchback_experiment(true_probs, n_hours=40, n_select=200, seed=42):
    """
    Run switchback experiment alternating Random and TS every hour.
    
    Returns:
        hourly_data: list of dicts with hour info
    """
    rng = np.random.default_rng(seed)
    n_dashers = len(true_probs)
    
    # Thompson Sampling priors: Beta(1, 1) for each dasher
    alpha_params = np.ones(n_dashers)
    beta_params = np.ones(n_dashers)
    
    hourly_data = []
    
    for hour in range(n_hours):
        # Alternate: even hours = Random, odd hours = Thompson Sampling
        if hour % 2 == 0:
            method = "Random"
            selected, responses = random_messaging(true_probs, n_select, rng)
        else:
            method = "Thompson Sampling"
            selected, responses = thompson_sampling(
                alpha_params, beta_params, true_probs, n_select, rng
            )
        
        # Update Thompson Sampling posteriors (always update, even during random hours)
        # This simulates that DoorDash learns from ALL interactions
        alpha_params[selected] += responses
        beta_params[selected] += (1 - responses)
        
        conversion_rate = responses.mean()
        hourly_data.append({
            "hour": hour,
            "method": method,
            "selected": selected,
            "responses": responses,
            "conversion_rate": conversion_rate,
            "n_conversions": responses.sum(),
        })
    
    return hourly_data


def analyze_switchback(hourly_data):
    """
    Analyze switchback experiment results.
    Returns ATE, p-value, 95% CI.
    """
    random_rates = [d["conversion_rate"] for d in hourly_data if d["method"] == "Random"]
    ts_rates = [d["conversion_rate"] for d in hourly_data if d["method"] == "Thompson Sampling"]
    
    random_rates = np.array(random_rates)
    ts_rates = np.array(ts_rates)
    
    # Average Treatment Effect
    ate = np.mean(ts_rates) - np.mean(random_rates)
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(ts_rates, random_rates)
    
    # 95% confidence interval for the difference in means
    n1, n2 = len(ts_rates), len(random_rates)
    se = np.sqrt(np.var(ts_rates, ddof=1) / n1 + np.var(random_rates, ddof=1) / n2)
    t_crit = stats.t.ppf(0.975, df=min(n1, n2) - 1)
    ci_lower = ate - t_crit * se
    ci_upper = ate + t_crit * se
    
    return {
        "ate": ate,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean_random": np.mean(random_rates),
        "mean_ts": np.mean(ts_rates),
        "random_rates": random_rates,
        "ts_rates": ts_rates,
    }


def plot_results(hourly_data, analysis, display=False):
    """Plot switchback experiment results."""
    os.makedirs("figures", exist_ok=True)
    
    # Plot 1: Conversion rates over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    hours = [d["hour"] for d in hourly_data]
    rates = [d["conversion_rate"] for d in hourly_data]
    colors = ['blue' if d["method"] == "Random" else 'red' for d in hourly_data]
    
    ax1 = axes[0]
    for h, r, c, d in zip(hours, rates, colors, hourly_data):
        ax1.bar(h, r, color=c, alpha=0.7,
                label=d["method"] if h <= 1 else "")
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Conversion Rate')
    ax1.set_title('Switchback Experiment: Conversion Rates by Hour')
    ax1.axhline(y=analysis["mean_random"], color='blue', linestyle='--',
                alpha=0.5, label=f'Random Mean = {analysis["mean_random"]:.4f}')
    ax1.axhline(y=analysis["mean_ts"], color='red', linestyle='--',
                alpha=0.5, label=f'TS Mean = {analysis["mean_ts"]:.4f}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Comparison box plot
    ax2 = axes[1]
    bp = ax2.boxplot([analysis["random_rates"], analysis["ts_rates"]],
                     tick_labels=["Random Messaging", "Thompson Sampling"],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Conversion Rate')
    ax2.set_title(f'Conversion Rate Distribution\n'
                  f'ATE = {analysis["ate"]:.4f}, '
                  f'p-value = {analysis["p_value"]:.4f}, '
                  f'95% CI = [{analysis["ci_lower"]:.4f}, {analysis["ci_upper"]:.4f}]')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if display:
        plt.show()
    else:
        plt.savefig('figures/switchback_experiment.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    print("\nGenerating 1000 dashers...")
    X, true_probs = generate_dashers(n_dashers=1000, seed=42)
    print(f"  True acceptance probability range: [{true_probs.min():.4f}, {true_probs.max():.4f}]")
    print(f"  Mean true acceptance probability:   {true_probs.mean():.4f}")
    
    print("\nRunning switchback experiment (40 hours)...")
    hourly_data = run_switchback_experiment(true_probs, n_hours=40, n_select=200, seed=42)
    
    print(f"\n{'Hour':>4} {'Method':<20} {'Conv. Rate':>10} {'Conversions':>12}")
    print("-" * 50)
    for d in hourly_data:
        print(f"{d['hour']:>4} {d['method']:<20} {d['conversion_rate']:>10.4f} {d['n_conversions']:>12}")
    
    analysis = analyze_switchback(hourly_data)
    
    print(f"  Mean Random Conversion Rate:    {analysis['mean_random']:.4f}, Mean TS Conversion Rate: {analysis['mean_ts']:.4f}")
    print(f"  Average Treatment Effect (ATE): {analysis['ate']:.4f}, p-value: {analysis['p_value']:.4f}, 95% Confidence Interval: [{analysis['ci_lower']:.4f}, {analysis['ci_upper']:.4f}]")
    
    plot_results(hourly_data, analysis)

if __name__ == "__main__":
    main()
