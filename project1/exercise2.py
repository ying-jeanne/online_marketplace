import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

def demand_menu1(f, v_L, v_H, v_LH):
    # Demand for L: buyers in [v_L, v_LH) who prefer L
    if v_L < v_LH:
        DL = quad(f, v_L, v_LH)[0]
    else:
        DL = 0.0  # L is never preferred
    # Demand for H: buyers >= v_LH who can afford H
    # They must also have v >= v_H for positive utility from H
    lower_H = max(v_LH, v_H)
    DH = quad(f, lower_H, np.inf)[0]
    return DL, DH

def demand_menu2(f, v_H):
    DH = quad(f, v_H, np.inf)[0]
    return DH

def revenue_menu1(pL, DL, pH, DH):
    return pL * DL + pH * DH

def revenue_menu2(pH, DH):
    return pH * DH

# The density functions
def f1(v):
    return 0.5 * v ** (-1.5)

def f2(v):
    return 3.0 * v ** (-4.0)

def plot_density_functions(display=False):
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    # Compute Demand, and plotting the density functions on [1,3]
    vs = np.linspace(1.0, 3.0, 400)
    plt.figure()
    plt.plot(vs, [f1(v) for v in vs], label="f1(v) = 0.5 * v^-1.5")
    plt.plot(vs, [f2(v) for v in vs], label="f2(v) = 3 * v^-4")
    plt.title("Density functions on v in [1,3]")
    plt.xlabel("v")
    plt.ylabel("f(v)")
    plt.legend()
    plt.tight_layout()
    if display:
        plt.show()
    else: 
        plt.savefig(os.path.join("figures", "density_functions.png"))
        plt.close()

def find_thresholds(pL, qL, pH, qH):
    # v_L: threshold where buyer is indifferent between L and not buying
    v_L = pL / qL  # = 1.5/1 = 1.5
    if v_L < 1.0:
        v_L = 1.0
    # v_H: threshold where buyer is indifferent between H and not buying
    v_H = pH / qH  # = 4/2 = 2.0
    if v_H < 1.0:
        v_H = 1.0
    # v_LH: threshold where buyer is indifferent between L and H
    v_LH = (pH - pL) / (qH - qL)  # = (4-1.5)/(2-1) = 2.5
    if v_LH < 1.0:
        v_LH = 1.0
    return v_L, v_H, v_LH

def main():
    plot_density_functions()

    # find the threshold v where user would chose between menus
    pL, qL, pH, qH = 1.5, 1.0, 4.0, 2.0
    v_L, v_H, v_LH = find_thresholds(pL, qL, pH, qH)

    demand_menu1_f1 = demand_menu1(f1, v_L, v_H, v_LH)
    demand_menu2_f1 = demand_menu2(f1, v_H)

    demand_menu1_f2 = demand_menu1(f2, v_L, v_H, v_LH)
    demand_menu2_f2 = demand_menu2(f2, v_H)

    print(f"\nThe demand results for \nf1 with menu 1 is: DL = {demand_menu1_f1[0]:.4f}, DH = {demand_menu1_f1[1]:.4f}. \nf1 with menu 2 is: DH = {demand_menu2_f1:.4f}. \nf2 with menu 1 is: DL = {demand_menu1_f2[0]:.4f}, DH = {demand_menu1_f2[1]:.4f}. \nf2 with menu 2 is: DH = {demand_menu2_f2:.4f}.\n")

    revenue_f1_menu1 = revenue_menu1(pL, demand_menu1_f1[0], pH, demand_menu1_f1[1])
    revenue_f1_menu2 = revenue_menu2(pH, demand_menu2_f1)

    revenue_f2_menu1 = revenue_menu1(pL, demand_menu1_f2[0], pH, demand_menu1_f2[1])
    revenue_f2_menu2 = revenue_menu2(pH, demand_menu2_f2)

    print(f"The revenue results for \nf1 with menu 1 is: R = {revenue_f1_menu1:.4f}. \nf1 with menu 2 is: R = {revenue_f1_menu2:.4f}. \nf2 with menu 1 is: R = {revenue_f2_menu1:.4f}. \nf2 with menu 2 is: R = {revenue_f2_menu2:.4f}.\n")

if __name__ == "__main__":
    main()