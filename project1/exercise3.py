import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
# create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

def get_trip_length(lambd=0.1, n_samples=10000, display_plot=False):
    mean_trip = 1 / lambd
    trip_lengths = np.random.exponential(scale=mean_trip, size=n_samples)
    _, _, _ = plt.hist(trip_lengths, bins=50, density=True, alpha=0.6, color='b', label='Simulated Data')

    # Plot the theoretical PDF as baseline f(x) = lambda * exp(-lambda * x)
    x = np.linspace(0, max(trip_lengths), 1000)
    pdf = lambd * np.exp(-lambd * x)
    plt.plot(x, pdf, linewidth=2, color='r', label='Theoretical PDF ($f(x) = 0.1 e^{-0.1x}$)')

    plt.title('Distribution of Trip Lengths (Exponential, lambda=0.1)')
    plt.xlabel('Trip Length (miles)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Sanity check
    print(f"Sample Mean: {np.mean(trip_lengths):.2f} (Target: 10.00)")
    if display_plot:
        plt.show()
    else:
        plt.savefig(os.path.join('figures', 'trip_length_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return trip_lengths

def compare_pricing_schemes(n_trips=10000, lambd=0.1, c=2.50, r=0.75, surge_bonus=5.00, display_plot=False):
    # 1. Generate Trip Lengths
    trip_lengths = get_trip_length(lambd, n_trips, display_plot)

    # 2. Generate Surge Multipliers with probability given
    # Values: {1.0, 1.5, 2.0, 2.5}
    # Probs:  {0.5, 0.3, 0.15, 0.05}
    multipliers = np.random.choice(
        [1.0, 1.5, 2.0, 2.5], 
        size=n_trips, 
        p=[0.5, 0.3, 0.15, 0.05]
    )

    # Base Fare (Cost)
    base_fares = c + r * trip_lengths

    # Scheme A: Multiplicative Surge
    payouts_mult = multipliers * base_fares

    # Scheme B: Additive Surge (Bonus)
    payouts_add = base_fares + (multipliers - 1) * surge_bonus

    # Analysis mean and variance of each Schema
    mean_mult = np.mean(payouts_mult)
    var_mult = np.var(payouts_mult)

    mean_add = np.mean(payouts_add)
    var_add = np.var(payouts_add)

    print(f"For Mulplicative Surge Pricing: Mean Payout: ${mean_mult:.2f}, Variance: {var_mult:.2f}")
    print(f"For Additive Surge Pricing: Mean Payout: ${mean_add:.2f}, Variance: {var_add:.2f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.hist(payouts_mult, bins=50, alpha=0.5, label='Multiplicative', color='blue', edgecolor='black')
    plt.hist(payouts_add, bins=50, alpha=0.5, label='Additive (Bonus)', color='orange', edgecolor='black')

    plt.title('Distribution of Payouts: Multiplicative vs Additive Surge')
    plt.xlabel('Payout Amount ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    if display_plot:
        plt.show()
    else:
        plt.savefig(os.path.join('figures', 'pricing_scheme_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    return payouts_mult, payouts_add

if __name__ == "__main__":
    # Compare pricing schemes and save the figure
    compare_pricing_schemes(display_plot=False)