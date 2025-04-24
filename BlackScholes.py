import scipy.stats as si
import numpy as np
import matplotlib.pyplot as plt

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calcule le prix d'une option européenne ainsi que les greeks via le modèle Black-Scholes.
    
    S : Prix actuel du sous-jacent
    K : Prix d'exercice
    T : Temps jusqu'à expiration (en années)
    r : Taux d'intérêt sans risque
    sigma : Volatilité
    option_type : "call" ou "put"
    
    Retourne : dictionnaire avec le prix et les greeks
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        delta = si.norm.cdf(d1)
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * si.norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        delta = -si.norm.cdf(-d1)
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * si.norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T)

    return {
        "x": 0,
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,   # vega est souvent exprimé par 1% de variation de sigma
        "theta": theta / 365, # theta par jour
        "rho": rho / 100      # rho est souvent exprimé pour 1% de variation de r
    }

def greeks_over_x(S_values, K, T, r, sigma, option_type="call"):
    greeks = {
        "x": S_values,
        "delta": [],
        "gamma": [],
        "vega": [],
        "theta": [],
        "rho": []
    }
    
    for S_sim in S_values:
        option = black_scholes(S_sim, K, T, r, sigma, option_type)
        greeks["delta"].append(option["delta"])
        greeks["gamma"].append(option["gamma"])
        greeks["vega"].append(option["vega"])
        greeks["theta"].append(option["theta"])
        greeks["rho"].append(option["rho"])
        
    return greeks

def plot_greeks(greeks_data, x_label="X", title_prefix=""):
    x = greeks_data["x"]
    
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(x, greeks_data["delta"])
    plt.title(f"{title_prefix}Delta")
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(x, greeks_data["gamma"], color="orange")
    plt.title(f"{title_prefix}Gamma")
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(x, greeks_data["vega"], color="green")
    plt.title(f"{title_prefix}Vega")
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(x, greeks_data["theta"], color="red")
    plt.title(f"{title_prefix}Theta")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(x, greeks_data["rho"], color="purple")
    plt.title(f"{title_prefix}Rho")
    plt.grid(True)

    plt.suptitle(f"Greeks en fonction de {x_label}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
