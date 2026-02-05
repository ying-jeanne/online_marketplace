import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def demand_A(p):
    return max(0, 100 - 10 * p)

def demand_B(p):
    return max(0, 50 - 8 * p)

# Objective functions for minimization (negative profit)
def neg_profit_A(p, supply_A):
    d = demand_A(p)
    q = min(d, supply_A)
    return -(p * q)

def neg_profit_B(p, supply_B):
    d = demand_B(p)
    q = min(d, supply_B)
    return -(p * q)

def neg_total_profit_uniform(p, supply_A=45, supply_B=35):
    return neg_profit_A(p, supply_A) + neg_profit_B(p, supply_B)

def optimize_spatial(supply_A=45, supply_B=35):
    # Optimize Zone A (Demand A becomes 0 at p=10)
    res_A = minimize_scalar(neg_profit_A, bounds=(0, 10), method='bounded', args=(supply_A,))
    opt_p_A = res_A.x
    opt_rev_A = -res_A.fun
    
    # Optimize Zone B (Demand B becomes 0 at p=6.25)
    res_B = minimize_scalar(neg_profit_B, bounds=(0, 6.25), method='bounded', args=(supply_B,))
    opt_p_B = res_B.x
    opt_rev_B = -res_B.fun
    
    total_profit = opt_rev_A + opt_rev_B
    sales_A = min(demand_A(opt_p_A), supply_A)
    sales_B = min(demand_B(opt_p_B), supply_B)
    print(f"Optimal Price Zone A: ${opt_p_A:.2f}, Sales Zone A: {sales_A:.2f}, Optimal Revenue Zone A: ${opt_rev_A:.2f}")
    print(f"Optimal Price Zone B: ${opt_p_B:.2f}, Sales Zone B: {sales_B:.2f}, Optimal Revenue Zone B: ${opt_rev_B:.2f}")
    print(f"Total Profit: ${total_profit:.2f}")

def optimize_uniform(display=False):
    res = minimize_scalar(neg_total_profit_uniform, bounds=(0, 10), method='bounded')
    opt_p = res.x
    opt_rev = -res.fun
    sales = min(demand_A(opt_p), 45) + min(demand_B(opt_p), 35)
    print(f"Optimal Uniform Price: ${opt_p:.2f}, Sales at Uniform price: {sales:.2f}, Total Profit: ${opt_rev:.2f}")
    prices = np.linspace(0, 12, 500)
    profits = [-neg_total_profit_uniform(p) for p in prices]

    plt.plot(prices, profits, label='Total Profit (Uniform Pricing)', color='blue')
    plt.title('Profit vs Price (Uniform Pricing)')
    plt.xlabel('Price ($)')
    plt.ylabel('Total Profit ($)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Highlight optimal price
    opt_profit = -neg_total_profit_uniform(opt_p)
    plt.axvline(x=opt_p, color='red', linestyle='--', label=f'Optimal Price (${opt_p:.2f})')
    plt.scatter([opt_p], [opt_profit], color='red')
    plt.text(opt_p, opt_profit + 5, f'${opt_profit:.2f}', ha='center', color='red')

    plt.legend()
    if display:
        plt.show()
    else:
        plt.savefig('figures/uniform_pricing_profit.png', dpi=300, bbox_inches='tight')
        plt.close()

def train_demand_models_with_rf():
    data_sample = pd.read_csv("./historical_data_with_demand_v2.csv")
    features_A = ["price_A", "market_sentiment", "zone_specific_A", "competitor_prices_A"]
    features_B = ["price_B", "market_sentiment", "zone_specific_B", "competitor_prices_B"]
    X_A = data_sample[features_A]
    y_A = data_sample["demand_A"]
    X_B = data_sample[features_B]
    y_B = data_sample["demand_B"]
    # Train-test split
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
    # Random Forest Regressor
    rf_A = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_B = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_A.fit(X_train_A, y_train_A)
    rf_B.fit(X_train_B, y_train_B)

    score_A = rf_A.score(X_test_A, y_test_A)
    score_B = rf_B.score(X_test_B, y_test_B)
    print(f"\nRandom Forest Model Performance: Zone A R^2 Score: {score_A:.3f}, Zone B R^2 Score: {score_B:.3f}.\n")
    return rf_A, rf_B

def get_demand_A_rf(rf_A, price, market_context):
    # Create a DataFrame with the exact feature names used in training
    # features_A = ["price_A", "market_sentiment", "zone_specific_A", "competitor_prices_A"]
    X_pred = pd.DataFrame({
        "price_A": [price],
        "market_sentiment": [market_context["market_sentiment"]],
        "zone_specific_A": [market_context["zone_specific_A"]],
        "competitor_prices_A": [market_context["competitor_prices_A"]]
    })
    # rf_A is the trained model from previous cell
    predicted_demand = rf_A.predict(X_pred)[0]
    return max(0, predicted_demand)

def get_demand_B_rf(rf_B, price, market_context):
    # features_B = ["price_B", "market_sentiment", "zone_specific_B", "competitor_prices_B"]
    X_pred = pd.DataFrame({
        "price_B": [price],
        "market_sentiment": [market_context["market_sentiment"]],
        "zone_specific_B": [market_context["zone_specific_B"]],
        "competitor_prices_B": [market_context["competitor_prices_B"]]
    })
    predicted_demand = rf_B.predict(X_pred)[0]
    return max(0, predicted_demand)

def optimize_spatial_rf(rf_A, rf_B, price_grid, market_context, S_A=45, S_B=45):
    # ---------------- Optimization (Grid Search) ----------------
    # 1. Spatial Pricing (Optimize independently)
    best_pA = 0
    max_rev_A = 0
    max_demand_A = 0
    for p in price_grid:
        d = get_demand_A_rf(rf_A, p, market_context)
        rev = p * min(d, S_A)
        if rev > max_rev_A:
            max_rev_A = rev
            best_pA = p
            max_demand_A = d

    best_pB = 0
    max_rev_B = 0
    max_demand_B = 0
    for p in price_grid:
        d = get_demand_B_rf(rf_B, p, market_context)
        rev = p * min(d, S_B)
        if rev > max_rev_B:
            max_rev_B = rev
            best_pB = p
            max_demand_B = d

    print(f"Optimal Price Zone A: ${best_pA:.2f}, Demand A at optimal price: {max_demand_A:.2f}, Optimal Revenue Zone A: ${max_rev_A:.2f}")
    print(f"Optimal Price Zone B: ${best_pB:.2f}, Demand B at optimal price: {max_demand_B:.2f}, Optimal Revenue Zone B: ${max_rev_B:.2f}")
    print(f"Total Profit: ${max_rev_A + max_rev_B:.2f}")

def optimize_uniform_rf(rf_A, rf_B, price_grid, market_context, S_A=45, S_B=45, display=False):
    # Uniform Pricing (Optimize pA = pB = p)
    best_p_uniform = 0
    max_rev_uniform = 0
    max_sales_A_uniform = 0
    max_sales_B_uniform = 0
    uniform_profits = []

    for p in price_grid:
        a = min(get_demand_A_rf(rf_A, p, market_context), S_A)
        b = min(get_demand_B_rf(rf_B, p, market_context), S_B)
        prof = p * a + p * b
        uniform_profits.append(prof)
        if prof > max_rev_uniform:
            max_rev_uniform = prof
            best_p_uniform = p
            max_sales_A_uniform = a
            max_sales_B_uniform = b

    print(f"Optimal Price: ${best_p_uniform:.2f}, Sales at optimal price: {max_sales_A_uniform + max_sales_B_uniform:.2f}, Total Profit: ${max_rev_uniform:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(price_grid, uniform_profits, label='Total Profit (Uniform)', color='purple')
    plt.axvline(x=best_p_uniform, color='red', linestyle='--', label=f'Optimal Price (${best_p_uniform:.2f})')
    plt.scatter([best_p_uniform], [max_rev_uniform], color='red', zorder=5)
    plt.text(best_p_uniform, max_rev_uniform + 5, f"${max_rev_uniform:.2f}", ha='center', color='black', fontdict={'weight': 'bold'})

    plt.title('Profit vs Price (Uniform Pricing) - RF Estimation')
    plt.xlabel('Price ($)')
    plt.ylabel('Total Profit ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    if display:
        plt.show()
    else:
        plt.savefig('figures/uniform_pricing_rf_profit.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Define market context
    market_context = {
        "market_sentiment": 1.0,
        "zone_specific_A": 2.1,
        "zone_specific_B": 1.0,
        "competitor_prices_A": 4.0,
        "competitor_prices_B": 6.0
    }

    # Profit optimization with known demand functions
    optimize_spatial()
    optimize_uniform()

    # Profit optimization with Random Forest demand estimation
    rf_A, rf_B = train_demand_models_with_rf()
    price_grid = np.linspace(0, 15, 151)  # Check every $0.10 from $0 to $15
    optimize_spatial_rf(rf_A, rf_B, price_grid, market_context)
    optimize_uniform_rf(rf_A, rf_B, price_grid, market_context, display=False)

if __name__ == "__main__":
    main()