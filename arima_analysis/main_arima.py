import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # Backend non-interactiv
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Create results directory
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

def test_stationarity(timeseries, name="Seria"):
    print(f"\n--- Testul ADF pentru {name} ---")
    result = adfuller(timeseries)
    print(f'Statistica ADF: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print("Concluzie: Seria este STATIONARA")
        return True
    else:
        print("Concluzie: Seria este NESTATIONARA")
        return False

# 1. Incarcare si Preprocesare Date
df_raw = pd.read_csv("date/somaj_bim.csv")
df = df_raw.copy()
df.columns = [col.strip() for col in df.columns]
df = df[["Luni", "Valoare"]]
df.columns = ["date", "valoare"]
df["date"] = df["date"].apply(transforma_luna_romaneasca)
df["valoare"] = df["valoare"].astype(str).str.replace(",", ".", regex=False).astype(float)
df = df.dropna().set_index("date").sort_index().asfreq("MS")

serie = df["valoare"]

# 2. Analiza Vizuala si Diferentiere
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(serie, label='Seria Originală (Nestaționară)')
plt.title('Rata Șomajului BIM - Nivel Original')
plt.legend()

# Prima diferenta
serie_diff = serie.diff().dropna()
plt.subplot(2, 1, 2)
plt.plot(serie_diff, color='orange', label='Prima Diferență (Staționară)')
plt.title('Rata Șomajului BIM - Prima Diferență (d=1)')
plt.legend()
plt.tight_layout()
plt.savefig("rezultate/nivele_vs_diferenta.png", dpi=300)

# 3. Teste Formale de Radacina Unitate (ADF)
test_stationarity(serie, "Seria Originala")
is_stationary_diff = test_stationarity(serie_diff, "Prima Diferenta")

# Ordinul de integrare
d = 1 if not test_stationarity(serie, "Seria pentru determinare d") else 0

# 4. Identificare (ACF & PACF pe seria diferentiata)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(serie_diff, ax=ax1, lags=40, title='ACF pe seria diferențiată - Identificare MA(q)')
plot_pacf(serie_diff, ax=ax2, lags=40, title='PACF pe seria diferențiată - Identificare AR(p)')
ax1.set_xlabel('Lag')
ax2.set_xlabel('Lag')
plt.tight_layout()
plt.savefig("rezultate/acf_pacf_arima.png", dpi=300)

# 5. Selectia Modelului ARIMA (p, d, q)
print(f"\n--- Selectia Modelului ARIMA (p, {d}, q) ---")
best_aic = np.inf
best_order = None
best_model_fit = None

for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(serie, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
                best_model_fit = results
        except:
            continue

print(f"Cel mai bun model ARIMA gasit: ARIMA{best_order} cu AIC={best_aic:.2f}")

# 6. Diagnosticul Modelului (Verificare Box-Jenkins)
print("\n--- Diagnosticul Reziduurilor (Romanian Labels) ---")
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(221)
ax1.plot(best_model_fit.resid)
ax1.set_title('Reziduuri Standardizate')
ax1.axhline(0, color='black', linestyle='--')

ax2 = fig.add_subplot(222)
ax2.hist(best_model_fit.resid, bins=20, density=True, alpha=0.6, color='g')
mu, sigma = np.mean(best_model_fit.resid), np.std(best_model_fit.resid)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
p_dist = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
ax2.plot(x, p_dist, 'k', linewidth=2)
ax2.set_title('Histogramă Reziduuri')

ax3 = fig.add_subplot(223)
sm.graphics.qqplot(best_model_fit.resid, line='s', ax=ax3)
ax3.set_title('Grafic Q-Q (Normalitate)')

ax4 = fig.add_subplot(224)
plot_acf(best_model_fit.resid, ax=ax4, title='ACF Reziduuri (Zgomot Alb)')
plt.tight_layout()
plt.savefig("rezultate/diagnostic_arima.png", dpi=300)

# Test Ljung-Box
lb = acorr_ljungbox(best_model_fit.resid, lags=[10], return_df=True)
print("\nTestul Ljung-Box (p-value):")
print(lb)

# 7. Prognoza cu intervale de incredere
train = serie.iloc[:-12]
test = serie.iloc[-12:]
model_final = ARIMA(train, order=best_order)
model_final_fit = model_final.fit()

forecast_res = model_final_fit.get_forecast(steps=12)
forecast_mean = forecast_res.summary_frame()["mean"]
ci = forecast_res.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Istoric (Train)')
plt.plot(test.index, test, label='Real (Test)', color='green')
plt.plot(forecast_mean.index, forecast_mean, label=f'Prognoză ARIMA{best_order}', color='red')
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='pink', alpha=0.3, label='IC 95%')
plt.title(f'Prognoză ARIMA - Rata Șomajului BIM')
plt.xlabel('An')
plt.ylabel('Valoare (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("rezultate/prognoza_arima_finala.png", dpi=300)

print("\n--- Analiza ARIMA Finalizata! ---")
