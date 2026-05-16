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
plot_acf(serie, ax=ax1, lags=40, title='Autocorelatie (ACF) - Identificare MA(q)')
plot_pacf(serie, ax=ax2, lags=40, title='Autocorelatie Partiala (PACF) - Identificare AR(p)')
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
best_model_fit.plot_diagnostics(figsize=(12, 10))
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
plt.plot(train.index, train, label="Date Antrenare", color="blue")
plt.plot(test.index, test, label="Date Reale (Test)", color="green", alpha=0.6)
plt.plot(forecast_mean.index, forecast_mean, label=f"Prognoza ARIMA{best_order}", color="red", linestyle="--")
plt.fill_between(confidence_intervals.index, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], color='pink', alpha=0.3, label="Interval Incredere 95%")

plt.title(f"Prognoza Serii de Timp - Model ARIMA{best_order}")
plt.xlabel("An")
plt.ylabel("Valoare")
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
