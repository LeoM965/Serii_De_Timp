import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

df = pd.read_csv("date_unite.csv")
df.columns = df.columns.str.strip()
df["Data"] = df["Data"].str.replace("Luna ", "", case=False).str.strip()

luni_ro = {
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
    "decembrie": "12",
}

def transforma_data(text):
    parti = text.split()  # Împarte textul în ['ianuarie', '2010']
    luna_text = parti[0].lower()  # Luăm numele lunii cu litere mici
    an = parti[1]  # Luăm anul
    luna_numar = luni_ro[luna_text]  # Căutăm numărul lunii în dicționar
    return f"{an}-{luna_numar}-01"  # Returnăm formatul standard YYYY-MM-DD

# Aplicăm funcția de transformare pe întreaga coloană
df["Data"] = df["Data"].apply(transforma_data)

# 4. Transformăm coloana în format oficial datetime și o setăm ca INDEX
df["Data"] = pd.to_datetime(df["Data"])
df.set_index("Data", inplace=True)

# Îi spunem explicit librăriei că datele au frecvență lunară (Monthly Start)
df.index.freq = "MS"

# Verificăm rezultatul
print("Datele au fost încărcate cu succes:")
print(df.head())

print("--- TESTUL DE COINTEGRARE JOHANSEN ---")
# Det_order = -1 (fara trend deterministic), k_ar_diff = 1 (numarul de lag-uri in diferente)
rezultat_johansen = coint_johansen(df, det_order=-1, k_ar_diff=1)

# Extragem statistica Trace și valorile critice (la nivelul de 5%)
print("Statistica Trace:", rezultat_johansen.lr1)
print("Valori critice (90%, 95%, 99%):")
print(rezultat_johansen.cvt)
# Interpretare: Daca Statistica Trace > Valoarea critica la 95% (coloana din mijloc), respingem ipoteza nula de non-cointegrare.

# Diferențiem datele pentru a le face staționare (obligatoriu pentru VAR standard)
df_diff = df.diff().dropna()

print("\n--- SELECTIA LAG-ULUI OPTIM PENTRU VAR ---")
model = VAR(df_diff)
# Selectam numarul optim de lag-uri (luni) pe baza criteriului AIC
rezultat_lag = model.select_order(maxlags=12)
print(rezultat_lag.summary())

# Extragem lag-ul optim recomandat de AIC
lag_optim = rezultat_lag.aic
print(f"\nLag-ul optim ales este: {lag_optim}")

# Antrenăm modelul cu lag-ul ales
rezultate_var = model.fit(lag_optim)
print(rezultate_var.summary())

print("\n--- TESTUL DE CAUZALITATE GRANGER ---")
# Testăm dacă Inflația cauzează Șomajul
print("Cauzalitate: Inflația -> Șomaj")
granger_1 = grangercausalitytests(df_diff[['Somaj', 'Inflatie']], maxlag=[lag_optim])

# Testăm dacă Șomajul cauzează Inflația
print("\nCauzalitate: Șomaj -> Inflație")
granger_2 = grangercausalitytests(df_diff[['Inflatie', 'Somaj']], maxlag=[lag_optim])
# Interpretare: Te uiți la p-value (ex: ssr based F test). Dacă p-value < 0.05, există cauzalitate Granger!

# Generăm IRF pe o perioadă de 12 luni
irf = rezultate_var.irf(12)

# Plotăm IRF
fig_irf = irf.plot(orth=True) # orth=True folosește descompunerea Cholesky (standard în macro)
plt.suptitle('Funcția de Răspuns la Impuls (IRF)', fontsize=14, y=1.05)
plt.show()

# Generăm și plotăm Descompunerea Varianței (FEVD)
fevd = rezultate_var.fevd(12)
fig_fevd = fevd.plot()
plt.suptitle('Descompunerea Varianței', fontsize=14, y=1.05)
plt.show()