import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error, mean_squared_error



def calcul_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def transforma_luna_romaneasca(text):
    luni = {
        "ianuarie": "01",
        "februarie": "02",
        "martie": "03",
        "aprilie": "04",
        "mai": "05",
        "iunie": "06",
        "iulie": "07",
        "august": "08",
        "septembrie": "09",
        "octombrie": "10",
        "noiembrie": "11",
        "decembrie": "12"
    }

    text = str(text).strip().lower()
    text = text.replace("luna", "").strip()

    parti = text.split()
    luna_text = parti[0]
    an = parti[1]

    luna_numar = luni[luna_text]

    return pd.to_datetime(f"{an}-{luna_numar}-01")


df_raw = pd.read_csv("date/somaj_bim.csv")

print("Coloane initiale:")
print(df_raw.columns)
print(df_raw.head())

df = df_raw.copy()
df.columns = [col.strip() for col in df.columns]

df = df[["Luni", "Valoare"]]
df.columns = ["date", "somaj_bim_procente"]

df["date"] = df["date"].apply(transforma_luna_romaneasca)

df["somaj_bim_procente"] = (
    df["somaj_bim_procente"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

df["somaj_bim_procente"] = pd.to_numeric(df["somaj_bim_procente"], errors="coerce")

df = df.dropna()
df = df.set_index("date")
df = df.sort_index()
df = df.asfreq("MS")

serie = df["somaj_bim_procente"]

train = serie.loc["2010-01-01":"2023-12-01"]
test = serie.loc["2024-01-01":"2025-12-01"]

print("\nNumar observatii train:", len(train))
print("Numar observatii test:", len(test))

print("\nPrimele valori:")
print(serie.head())

print("\nUltimele valori:")
print(serie.tail())


plt.figure(figsize=(12, 5))
plt.plot(train, label="Train 2010-2023")
plt.plot(test, label="Test 2024-2025")
plt.title("Rata somajului BIM - impartire train/test")
plt.xlabel("An")
plt.ylabel("Rata somajului BIM (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("impartire_train_test.png", dpi=300)
plt.show()


model_ses = SimpleExpSmoothing(
    train,
    initialization_method="estimated"
)

fit_ses = model_ses.fit(optimized=True)
forecast_ses = fit_ses.forecast(len(test))


model_holt = Holt(
    train,
    initialization_method="estimated"
)

fit_holt = model_holt.fit(optimized=True)
forecast_holt = fit_holt.forecast(len(test))


model_hw_add = ExponentialSmoothing(
    train,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
)

fit_hw_add = model_hw_add.fit(optimized=True)
forecast_hw_add = fit_hw_add.forecast(len(test))


model_hw_mul = ExponentialSmoothing(
    train,
    trend=None,
    seasonal="mul",
    seasonal_periods=12,
    initialization_method="estimated"
)

fit_hw_mul = model_hw_mul.fit(optimized=True)
forecast_hw_mul = fit_hw_mul.forecast(len(test))


rezultate = []

modele = {
    "Simple Exponential Smoothing": forecast_ses,
    "Holt": forecast_holt,
    "Holt-Winters aditiv": forecast_hw_add,
    "Holt-Winters multiplicativ": forecast_hw_mul
}

for nume_model, predictii in modele.items():
    mae = mean_absolute_error(test, predictii)
    rmse = np.sqrt(mean_squared_error(test, predictii))
    mape = calcul_mape(test, predictii)

    rezultate.append({
        "Model": nume_model,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4)
    })

rezultate_df = pd.DataFrame(rezultate)

print("\nErori de prognoza:")
print(rezultate_df)

rezultate_df.to_csv("rezultate_holt_winters.csv", index=False)


prognoza_df = pd.DataFrame({
    "real": test,
    "simple_exponential_smoothing": forecast_ses,
    "holt": forecast_holt,
    "holt_winters_aditiv": forecast_hw_add,
    "holt_winters_multiplicativ": forecast_hw_mul
})

prognoza_df.to_csv("prognoze_holt_winters.csv")

print("\nPrognoze:")
print(prognoza_df)


# Cerinta 5: Interval de incredere folosind ETSModel (pentru Holt-Winters Multiplicativ)
model_ets = ETSModel(train, error="add", trend="add", seasonal="add", seasonal_periods=12)
fit_ets = model_ets.fit(disp=False)
pred = fit_ets.get_prediction(start=test.index[0], end=test.index[-1])
pred_df = pred.summary_frame(alpha=0.05) # Interval de incredere de 95%

plt.figure(figsize=(12, 6))
plt.plot(train, label="Set de Antrenare (Train: 2010-2023)", color="blue")
plt.plot(test, label="Set de Validare (Test: 2024-2025)", color="green")
plt.plot(pred_df['mean'], label="Prognoza Punctuala (Holt-Winters)", color="red", linestyle="--")
plt.fill_between(pred_df.index, pred_df['pi_lower'], pred_df['pi_upper'], color='pink', alpha=0.3, label="Interval de Incredere 95%")
plt.title("Cerinta 5: Prognoza ratei somajului BIM - Holt-Winters")
plt.xlabel("An")
plt.ylabel("Rata somajului BIM (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("grafic_holt_winters_prognoza.png", dpi=300)



plt.figure(figsize=(12, 5))
plt.plot(test, marker="o", label="Valori reale")
plt.plot(forecast_ses, marker="o", label="Simple Exponential Smoothing")
plt.plot(forecast_holt, marker="o", label="Holt")
plt.plot(forecast_hw_add, marker="o", label="Holt-Winters aditiv")
plt.plot(forecast_hw_mul, marker="o", label="Holt-Winters multiplicativ")
plt.title("Compararea metodelor de netezire exponentiala pe setul de test")
plt.xlabel("An")
plt.ylabel("Rata somajului BIM (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparatie_netezire_exponentiala.png", dpi=300)
plt.show()