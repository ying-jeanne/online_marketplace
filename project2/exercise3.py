import numpy as np
import matplotlib.pyplot as plt
import os


# -- Retailer parameters (from Table 1) -----------------------------------
ALPHA = np.array([0.62, 0.7, 0.75])       # Base attractiveness
BETA  = np.array([3.32, 3.1, 3.55])       # Own price sensitivity
GAMMA_CROSS = np.array([1.5, 1.1, 1.22])  # Competitor price sensitivity
S     = np.array([4.6, 4.7, 4.5])         # Quality score
DELTA = np.array([0.15, 0.15, 0.2])       # Quality sensitivity
B_IND = np.array([0, 0, 1])               # Badge indicator
THETA = np.array([0.3, 0.3, 0.3])         # Badge effect

# Price options for retailers
PRICE_OPTIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def compute_purchase_probabilities(prices, b):
    """
    Compute purchase probabilities using multinomial logit model.
    
    Args:
        prices: array of shape (3,) -- prices set by each retailer
        b: buyer fee multiplier
    
    Returns:
        probs: array of shape (4,) -- [q0, q1, q2, q3]
    """
    V = np.zeros(3)
    effective_prices = prices * (1 + b)
    
    for i in range(3):
        cross_price_sum = sum(prices[j] * (1 + b) for j in range(3) if j != i)
        V[i] = (ALPHA[i]
                - BETA[i] * effective_prices[i]
                + GAMMA_CROSS[i] * cross_price_sum
                + DELTA[i] * S[i]
                + THETA[i] * B_IND[i])
    
    # V0 = 0 (outside option)
    exp_V = np.exp(V)
    denom = 1.0 + np.sum(exp_V)  # 1 for exp(V0)=exp(0)=1
    
    q0 = 1.0 / denom
    q = exp_V / denom
    
    return np.concatenate(([q0], q))


class Exp3Seller:
    """Exp3 algorithm for a single seller."""
    
    def __init__(self, n_actions, gamma, rng):
        self.n_actions = n_actions
        self.gamma = gamma
        self.rng = rng
        self.weights = np.ones(n_actions)
    
    def get_probabilities(self):
        """Compute action probabilities."""
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + self.gamma / self.n_actions
        return probs
    
    def choose_action(self):
        """Choose an action according to Exp3 probabilities."""
        probs = self.get_probabilities()
        action = self.rng.choice(self.n_actions, p=probs)
        return action
    
    def update(self, action, reward):
        """Update weights based on observed reward."""
        probs = self.get_probabilities()
        # Estimated reward (importance-weighted)
        estimated_reward = reward / probs[action]
        # Update weight for chosen action
        self.weights[action] *= np.exp(self.gamma * estimated_reward / self.n_actions)


def simulate_one_run(b, T, gamma_exp3, rng):
    """
    Run one simulation of T time steps.
    
    Returns:
        total_platform_revenue: total platform revenue over T steps
        total_q0: sum of q0 over T steps (for averaging later)
    """
    n_actions = len(PRICE_OPTIONS)
    sellers = [Exp3Seller(n_actions, gamma_exp3, rng) for _ in range(3)]
    
    total_platform_revenue = 0.0
    total_q0 = 0.0
    
    for t in range(T):
        # Each seller chooses a price action
        actions = [seller.choose_action() for seller in sellers]
        prices = PRICE_OPTIONS[actions]
        
        # Compute purchase probabilities
        probs = compute_purchase_probabilities(prices, b)
        q0 = probs[0]
        total_q0 += q0
        
        # Buyer makes purchase decision
        buyer_choice = rng.choice(4, p=probs)  # 0=no buy, 1-3=retailer
        
        if buyer_choice == 0:
            # Update sellers with 0 reward for their chosen actions
            for i, seller in enumerate(sellers):
                seller.update(actions[i], 0.0)
        else:
            # Sale to retailer (buyer_choice - 1)
            chosen_retailer = buyer_choice - 1
            chosen_price = prices[chosen_retailer]
            
            # Platform revenue = total fee = 10% of transaction price
            # The total fee is always 10%: buyer pays P*(1+b), seller receives P*(0.9+b)
            # Platform revenue = P*(1+b) - P*(0.9+b) = P*0.1
            platform_revenue = chosen_price * 0.10
            total_platform_revenue += platform_revenue
            
            # Seller revenue = price * (0.9 + b) -- what they receive
            # But for Exp3, we normalize reward to [0,1]
            for i, seller in enumerate(sellers):
                if i == chosen_retailer:
                    # Seller's reward: normalized revenue
                    # Seller receives P*(0.9+b), normalize by max possible: 1.0*(0.9+0.10)=1.0
                    seller_revenue = chosen_price * (0.9 + b)
                    reward = min(seller_revenue, 1.0)  # Cap at 1
                    seller.update(actions[i], reward)
                else:
                    seller.update(actions[i], 0.0)
    
    return total_platform_revenue, total_q0 / T


def run_all_simulations(b_values, T=1000, n_runs=200, gamma_exp3=0.2, base_seed=42):
    """
    Run simulations for all values of b.
    
    Returns:
        avg_revenues: dict b -> average platform revenue
        avg_q0s: dict b -> average q0
    """
    avg_revenues = {}
    avg_q0s = {}
    
    for b in b_values:
        revenues = []
        q0s = []
        
        for run in range(n_runs):
            rng = np.random.default_rng(base_seed + run)
            rev, q0 = simulate_one_run(b, T, gamma_exp3, rng)
            revenues.append(rev)
            q0s.append(q0)
        
        avg_revenues[b] = np.mean(revenues)
        avg_q0s[b] = np.mean(q0s)
        
        print(f"  b = {b:.2f}: avg revenue = {avg_revenues[b]:.2f}, avg q0 = {avg_q0s[b]:.4f}")
    
    return avg_revenues, avg_q0s


def plot_results(b_values, avg_revenues, avg_q0s, display=False):
    """Plot average platform revenue and q0 vs b."""
    os.makedirs("figures", exist_ok=True)
    
    rev_list = [avg_revenues[b] for b in b_values]
    q0_list = [avg_q0s[b] for b in b_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Revenue plot
    ax1.plot(b_values, rev_list, 'b-o', markersize=5)
    best_b_rev = b_values[np.argmax(rev_list)]
    ax1.axvline(x=best_b_rev, color='r', linestyle='--', alpha=0.7,
                label=f'Optimal b = {best_b_rev:.2f}')
    ax1.set_xlabel('Buyer Fee Multiplier (b)')
    ax1.set_ylabel('Average Platform Revenue ($)')
    ax1.set_title('Average Platform Revenue vs Buyer Fee (b)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # q0 plot
    ax2.plot(b_values, q0_list, 'r-o', markersize=5)
    best_b_q0 = b_values[np.argmin(q0_list)]
    ax2.axvline(x=best_b_q0, color='b', linestyle='--', alpha=0.7,
                label=f'Min q0 at b = {best_b_q0:.2f}')
    ax2.set_xlabel('Buyer Fee Multiplier (b)')
    ax2.set_ylabel('Average Probability of No Sale (q0)')
    ax2.set_title('Average No-Sale Probability vs Buyer Fee (b)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if display:
        plt.show()
    else:
        plt.savefig('figures/fee_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():    
    b_values = np.round(np.arange(0.00, 0.11, 0.01), 2)
    
    print(f"\nSimulating for b in {{{', '.join(f'{b:.2f}' for b in b_values)}}}")
    print(f"T = 1000, Runs = 200, gamma = 0.2\n")
    
    avg_revenues, avg_q0s = run_all_simulations(b_values, T=1000, n_runs=200)
    
    rev_list = [avg_revenues[b] for b in b_values]
    q0_list = [avg_q0s[b] for b in b_values]
    
    best_b_rev = b_values[np.argmax(rev_list)]
    best_b_q0 = b_values[np.argmin(q0_list)]
    
    print(f"Optimal b for max revenue:  {best_b_rev:.2f} (revenue = ${max(rev_list):.2f}), Optimal b for min q0: {best_b_q0:.2f} (q0 = {min(q0_list):.4f})")
    
    plot_results(b_values, avg_revenues, avg_q0s)


if __name__ == "__main__":
    main()
