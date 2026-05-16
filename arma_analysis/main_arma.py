import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # Utilizam un backend non-interactiv pentru a evita erorile de Tcl/Tkinter
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Create results directory if it doesn't exist
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

def test_stationarity(timeseries):
    print("\n--- Testul ADF (Augmented Dickey-Fuller) ---")
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    if result[1] <= 0.05:
        print("Concluzie: Seria este STATIOANARA (respingem H0)")
        return True
    else:
        print("Concluzie: Seria este NESTATIONARA (nu respingem H0)")
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

# 2. Verificarea Stationaritatii
# ARMA presupune stationaritate. Daca seria nu este stationara, ar trebui diferentiata (ARIMA),
# insa pentru acest exercitiu ne concentram pur pe ARMA (d=0).
test_stationarity(serie)

# 3. Identificare (ACF & PACF) - Semnatura ARMA
# AR: PACF se anuleaza dupa lag p
# MA: ACF se anuleaza dupa lag q
# ARMA: Ambele scad gradual (exponential sau sinusoidal amortizat)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(serie, ax=ax1, lags=40, title='Funcția de Autocorelație (ACF) - Identificare MA(q)')
plot_pacf(serie, ax=ax2, lags=40, title='Funcția de Autocorelație Parțială (PACF) - Identificare AR(p)')
ax1.set_xlabel('Lag (Întârziere)')
ax1.set_ylabel('Corelație')
ax2.set_xlabel('Lag (Întârziere)')
ax2.set_ylabel('Corelație Parțială')
plt.tight_layout()
plt.savefig("rezultate/acf_pacf_identificare.png", dpi=300)

# 4. Selectia Modelului ARMA (Grid Search pe p si q)
# d=0 pentru a ramane in spectrul ARMA
print("\n--- Selectia Modelului ARMA (p, 0, q) ---")
best_aic = np.inf
best_order = None
best_model_fit = None

# Cautam p si q conform principiului parcimoniei
for p in range(4): # Testam p pana la 3
    for q in range(4): # Testam q pana la 3
        try:
            # ARIMA(p, 0, q) este echivalent cu ARMA(p, q)
            model = ARIMA(serie, order=(p, 0, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, 0, q)
                best_model_fit = results
        except:
            continue

print(f"Cel mai bun model ARMA gasit: ARMA({best_order[0]}, {best_order[2]}) cu AIC={best_aic:.2f}")

# 5. Diagnosticul Modelului (Verificarea reziduurilor conform slide-urilor)
# Reziduurile trebuie sa fie Zgomot Alb (White Noise)
print("\n--- Diagnosticul Reziduurilor (Verificare Box-Jenkins) ---")

# Recreăm manual graficele de diagnostic pentru a avea titluri în Română
fig = plt.figure(figsize=(12, 10))

# 1. Reziduuri în timp
ax1 = fig.add_subplot(221)
ax1.plot(best_model_fit.resid)
ax1.set_title('Reziduuri Standardizate')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)

# 2. Histogramă + Densitate
ax2 = fig.add_subplot(222)
ax2.hist(best_model_fit.resid, bins=20, density=True, alpha=0.6, color='g')
# Adăugăm curba normală teoretică pentru referință
xmin, xmax = ax2.get_xlim()
x = np.linspace(xmin, xmax, 100)
mu = np.mean(best_model_fit.resid)
sigma = np.std(best_model_fit.resid)
p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
ax2.plot(x, p, 'k', linewidth=2, label='Normală')
ax2.set_title('Histogramă și Densitate')
ax2.set_xlabel('Eroare')
ax2.set_ylabel('Densitate')

# 3. Normal Q-Q Plot
ax3 = fig.add_subplot(223)
sm.graphics.qqplot(best_model_fit.resid, line='s', ax=ax3)
ax3.set_title('Grafic Normal Q-Q')
ax3.set_xlabel('Cuantile Teoretice')
ax3.set_ylabel('Cuantile Eșantion')

# 4. Corelograma reziduurilor
ax4 = fig.add_subplot(224)
plot_acf(best_model_fit.resid, ax=ax4, title='Corelograma Reziduurilor (ACF)')
ax4.set_xlabel('Lag')

plt.tight_layout()
plt.savefig("rezultate/diagnostic_model_arma.png", dpi=300)

# Testul Ljung-Box: H0 = Reziduurile sunt independente (zgomot alb)
lb_test = acorr_ljungbox(best_model_fit.resid, lags=[10], return_df=True)
print("\nTestul Ljung-Box:")
print(lb_test)

if lb_test['lb_pvalue'].iloc[0] > 0.05:
    print("Concluzie: Modelul este ADECVAT (reziduurile sunt zgomot alb)")
else:
    print("Concluzie: Modelul ar putea fi imbunatatit (exista autocorelatie in reziduuri)")

# 6. Invertibilitate si Stationaritate (Radacinile polinomului)
print("\n--- Verificare Radacini (Invertibilitate & Stationaritate) ---")
print("Radacini AR (trebuie sa fie > 1 pentru stationaritate):")
print(np.abs(best_model_fit.arroots))
print("Radacini MA (trebuie sa fie > 1 pentru invertibilitate):")
print(np.abs(best_model_fit.maroots))

# 7. Prognoza ARMA
# Impartire train/test
train = serie.iloc[:-12]
test = serie.iloc[-12:]

# Re-estimare pe datele de antrenare
model_final = ARIMA(train, order=best_order)
model_final_fit = model_final.fit()

# Prognoza
forecast_res = model_final_fit.get_forecast(steps=12)
forecast_mean = forecast_res.summary_frame()["mean"]
confidence_intervals = forecast_res.conf_int()

# 7. Vizualizare Rezultate Finale
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Date Antrenare (Istoric)", color="blue")
plt.plot(test.index, test, label="Date Reale (Validare)", color="green", alpha=0.6)
plt.plot(forecast_mean.index, forecast_mean, label=f"Prognoza ARMA({best_order[0]}, {best_order[2]})", color="red", linestyle="--")
plt.fill_between(confidence_intervals.index, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], color='pink', alpha=0.3, label="Interval de Încredere 95%")

plt.title(f"Prognoza Serii de Timp - Model ARMA({best_order[0]}, {best_order[2]})")
plt.xlabel("Anul")
plt.ylabel("Rata Șomajului BIM (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rezultate/prognoza_finala_arma.png", dpi=300)

# Calcul erori
mae = mean_absolute_error(test, forecast_mean)
rmse = np.sqrt(mean_squared_error(test, forecast_mean))
print(f"\nErori pe setul de test (12 luni):")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Salvare rezultate
results_summary = pd.DataFrame({
    "Data": test.index,
    "Real": test.values,
    "Prognoza": forecast_mean.values,
    "Lower_CI": confidence_intervals.iloc[:, 0].values,
    "Upper_CI": confidence_intervals.iloc[:, 1].values
})
results_summary.to_csv("rezultate/prognoze_detaliate_arma.csv", index=False)

print("\n--- Proces Finalizat cu Succes! ---")
print("Rezultatele au fost salvate in folderul 'rezultate/'.")
