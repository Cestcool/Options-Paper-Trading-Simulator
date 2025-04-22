import yfinance as yf
import numpy as np
from datetime import datetime
import re
from BlackScholes import black_scholes, greeks_over_x, plot_greeks

# ---- Paramètres de base ----
ticker_symbol = "AAPL"
start_date = "2025-03-20"
end_date = "2025-05-16"
option_type = "call"     # "call" ou "put"
strike_price = 195
premium = 8.55
contract_size = 100
risk_free_rate = 0.04  # Taux d'intérêt sans risque
vol_period = 252

noise = 0.01     # % de fluctation max de l'IV par rapport a la vol théorique

# ---- Téléchargement des données ----
data = yf.download(ticker_symbol, start=start_date, end=end_date)[['Close']]
data = data.reset_index()
data['Date'] = data['Date'].dt.date

# ---- Calcul de la volatilité historique ----
# Calcul de la volatilité (écart-type des rendements quotidiens)
data['Returns'] = data['Close'].pct_change()
volatility = data['Returns'].std() * np.sqrt(vol_period)                        # Revoir le calcule de la vol
# Initialisation de la volatilité implicite simulée
vol_imp = volatility  # Valeur de base, dérivée de la volatilité historique
vol_history = [vol_imp]

# ---- Simulation ----
print(f"\n🏁 Début de la simulation pour l'option {option_type.upper()} sur {ticker_symbol}")
print(f"Strike: {strike_price} USD | Premium: {premium} USD | Expire le: {end_date}")
print("Tu peux exercer, revendre ou laisser expirer ton option chaque jour.")

capital = -premium * contract_size  # coût initial
held = True
decision_taken = False

for i, row in data.iterrows():
    date = row['Date']
    # --- PAS PROPRE ---
    date_string = str(row['Date'])
    date_cleaned = re.search(r'\d{4}-\d{2}-\d{2}', date_string).group()
    # --- FIN DE PAS PROPRE ---
    close = float(row['Close'].iloc[0])
    
    # Variation aléatoire de la volatilité implicite (simulation réaliste)
    daily_noise = np.random.normal(0, noise)
    market_impact = 0.05 * data['Returns'].iloc[i] if i > 0 else 0  # Réaction aux rendements

    # Simulation d'une nouvelle volatilité implicite
    vol_imp = max(0.01, vol_imp * (1 + daily_noise + market_impact))  # Garde une vol minimale
    vol_history.append(vol_imp)
    
    # Calcul du temps restant jusqu'à expiration
    expiry_date = datetime.strptime(end_date, "%Y-%m-%d")
    current_date = datetime.strptime(date_cleaned, "%Y-%m-%d")
    time_to_expiry = (expiry_date - current_date).days / 365.0  # En années

    # Calcul du prix théorique de l'option via Black-Scholes
    option = black_scholes(close, strike_price, time_to_expiry, risk_free_rate, vol_imp, option_type)
    option_price = option["price"]

    print(f"\n📅 {date_cleaned} | Prix: {close:.2f} USD | Vol impl.: {vol_imp:.2%} | Valeur option (BS): {option_price:.2f} USD")

    # Dernier jour : forcer la décision
    if i == len(data) - 1:
        print("⏳ Dernier jour ! Option expirée.")
        decision_taken = True
        held = False
        capital += option_price * contract_size
        break

    # Si déjà revendue/exercée
    if not held:
        continue

    # Proposer une décision
    decision = input("💡 Que veux-tu faire ? [e]xercer, [r]evendre, [g]raphique, [n]e rien faire : ").strip().lower()

    if decision == "g":
        # Calcul de S_values autour du prix actuel S, avec une variation de +-5% du prix
        S_values = np.linspace(0.95 * close, 1.05 * close, 100)
        # Calcul des greeks pour cette gamme de valeurs de S
        greeks = greeks_over_x(S_values, strike_price, time_to_expiry, risk_free_rate, volatility, "call")
        plot_greeks(greeks)
        decision = input("💡 Que veux-tu faire ? [e]xercer, [r]evendre, [n]e rien faire : ").strip().lower()
        
    if decision =="i":
        print(volatility)

    if decision == "e":
        print("✅ Tu exerces l’option.")
        capital += option_price * contract_size
        held = False
        decision_taken = True
    elif decision == "r":
        print("💸 Tu revends l’option.")
        capital += option_price * contract_size
        held = False
        decision_taken = True
    elif decision == "n":
        print("🕒 Tu attends un meilleur moment.")
    else:
        print("Commande non reconnue, on passe au jour suivant.")

# Résumé final
print("\n--- RÉSULTAT FINAL ---")
print(f"Capital total : {capital:.2f} USD")
print("Gain" if capital > 0 else "Perte", "réalisé(e) 💰" if capital > 0 else "subie ❌")
