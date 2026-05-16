import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # Backend non-interactiv
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs("rezultate", exist_ok=True)

def transforma_luna_romaneasca(text):
    luni = {
        "ianuarie": "01", "februarie": "02", "martie": "03", "aprilie": "04",
        "mai": "05", "iunie": "06", "iulie": "07", "august": "08",
        "septembrie": "09", "octombrie": "10", "noiembrie": "11", "decembrie": "12"
    }
    text = str(text).strip().lower().replace("luna", "").strip()
    parti = text.split()
    if len(parti) < 2: return None
    luna_numar = luni.get(parti[0], "01")
    an = parti[1]
    return pd.to_datetime(f"{an}-{luna_numar}-01")

print("--- Modul SARIMA - Initiere Analiza ---")

# 1. Incarcare Date
df_raw = pd.read_csv("date/somaj_bim.csv")
df = df_raw.copy()
df.columns = [col.strip() for col in df.columns]
df = df[["Luni", "Valoare"]]
df.columns = ["date", "valoare"]
df["date"] = df["date"].apply(transforma_luna_romaneasca)
df["valoare"] = df["valoare"].astype(str).str.replace(",", ".", regex=False).astype(float)
df = df.dropna().set_index("date").sort_index().asfreq("MS")

serie = df["valoare"]

# 2. Transformare si Diferentiere (Sezonalitate Multiplicativa)
# Aplicam logaritm pentru a stabiliza varianta (echivalent Box-Cox lambda=0)
serie_log = np.log(serie)

# Diferenta obisnuita (d=1) si diferenta sezoniera (D=1, s=12)
serie_diff_reg = serie_log.diff().dropna()
serie_diff_both = serie_diff_reg.diff(12).dropna()

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(serie, label='Seria Originala')
plt.legend()
plt.title('Rata Somajului BIM')

plt.subplot(3, 1, 2)
plt.plot(serie_log, color='orange', label='Log(Serie)')
plt.legend()
plt.title('Transformare Logaritmica (Stabilizare Varianta)')

plt.subplot(3, 1, 3)
plt.plot(serie_diff_both, color='green', label='Log(Serie) - Dublu Diferentiata (d=1, D=1)')
plt.legend()
plt.title('Serie Stationara (Trend si Sezonalitate Eliminate)')
plt.tight_layout()
plt.savefig("rezultate/sarima_diferentieri.png", dpi=300)

# 3. Analiza ACF/PACF pentru identificare (p, q) si (P, Q)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(serie_diff_both, ax=ax1, lags=36, title='ACF Seria Dublu Diferentiata')
plot_pacf(serie_diff_both, ax=ax2, lags=36, title='PACF Seria Dublu Diferentiata')
plt.tight_layout()
plt.savefig("rezultate/sarima_acf_pacf.png", dpi=300)

# 4. Selectia Modelelor Candidate
print("\nEvaluare Modele SARIMA (AIC):")
modele_candidate = [
    ((0, 1, 1), (0, 1, 1, 12)), # Modelul Airline (baseline excelent)
    ((1, 1, 0), (1, 1, 0, 12)),
    ((1, 1, 1), (0, 1, 1, 12)),
    ((0, 1, 1), (1, 1, 1, 12))
]

best_aic = np.inf
best_model = None
best_fit = None

for order, seasonal_order in modele_candidate:
    try:
        model = SARIMAX(serie_log, order=order, seasonal_order=seasonal_order, 
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        print(f"SARIMA{order}x{seasonal_order} - AIC: {results.aic:.2f}")
        if results.aic < best_aic:
            best_aic = results.aic
            best_model = (order, seasonal_order)
            best_fit = results
    except Exception as e:
        continue

print(f"\nCel mai bun model: SARIMA{best_model[0]}x{best_model[1]} cu AIC={best_aic:.2f}")

# 5. Diagnosticul Reziduurilor
print("\n--- Diagnostic Reziduuri ---")
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(221)
ax1.plot(best_fit.resid)
ax1.set_title('Reziduuri Standardizate')
ax1.axhline(0, color='black', linestyle='--')

ax2 = fig.add_subplot(222)
ax2.hist(best_fit.resid, bins=20, density=True, alpha=0.6, color='g')
mu, sigma = np.mean(best_fit.resid), np.std(best_fit.resid)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
p_dist = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
ax2.plot(x, p_dist, 'k', linewidth=2)
ax2.set_title('Histograma si Densitate')

ax3 = fig.add_subplot(223)
sm.graphics.qqplot(best_fit.resid, line='s', ax=ax3)
ax3.set_title('Grafic Q-Q (Normalitate)')

ax4 = fig.add_subplot(224)
plot_acf(best_fit.resid, ax=ax4, title='Corelograma Reziduurilor (Zgomot Alb)', lags=24)
plt.tight_layout()
plt.savefig("rezultate/sarima_diagnostic.png", dpi=300)

# Ljung-Box Test
lb = acorr_ljungbox(best_fit.resid, lags=[12, 24], return_df=True)
print("\nTestul Ljung-Box (H0: Zgomot Alb):")
print(lb)

# 6. Prognoza Out-of-Sample
train = serie_log.iloc[:-12]
test = serie_log.iloc[-12:]

model_final = SARIMAX(train, order=best_model[0], seasonal_order=best_model[1],
                      enforce_stationarity=False, enforce_invertibility=False)
model_final_fit = model_final.fit(disp=False)

forecast_log = model_final_fit.get_forecast(steps=12)
forecast_mean_log = forecast_log.summary_frame()["mean"]
ci_log = forecast_log.conf_int()

# Transformare inversa (din logaritmic inapoi in nivel normal)
# Aplicam corectia de bias pentru transformarea exponentiala: exp(y + sigma^2 / 2)
varianta_estimata = np.var(model_final_fit.resid)
forecast_mean = np.exp(forecast_mean_log + (varianta_estimata / 2))
ci_lower = np.exp(ci_log.iloc[:, 0])
ci_upper = np.exp(ci_log.iloc[:, 1])

train_normal = np.exp(train)
test_normal = np.exp(test)

plt.figure(figsize=(12, 6))
plt.plot(train_normal.index, train_normal, label='Date Antrenare (Istoric)', color='blue')
plt.plot(test_normal.index, test_normal, label='Date Reale (Validare)', color='green')
plt.plot(forecast_mean.index, forecast_mean, label=f'Prognoza SARIMA', color='red', linestyle='--')
plt.fill_between(ci_lower.index, ci_lower, ci_upper, color='pink', alpha=0.3, label='Interval Incredere 95%')

plt.title(f"Prognoza Serii de Timp - Model SARIMA{best_model[0]}x{best_model[1]}")
plt.xlabel("Anul")
plt.ylabel("Rata Somajului BIM (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rezultate/sarima_prognoza_finala.png", dpi=300)

print("\n--- Analiza SARIMA Finalizata cu Succes! ---")
