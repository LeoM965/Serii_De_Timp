# Analiza și Prognoza Seriilor de Timp

Acest proiect reprezintă o abordare comprehensivă asupra analizei seriilor de timp, concentrându-se pe metode statistice avansate pentru prognoza indicatorilor macroeconomici (precum **Rata Șomajului BIM** și **Indicele Prețurilor de Consum - IPC**). 

Proiectul este împărțit în trei module principale, fiecare acoperind o metodologie distinctă din domeniul econometriei: **Netezirea Exponențială (Holt-Winters)**, modelarea **ARMA** (pentru serii staționare) și modelarea **ARIMA** (pentru serii nestaționare, conform metodologiei Box-Jenkins).

---

## 📁 Structura Proiectului

- `documents/`: Seturile de date inițiale în format CSV.
- `alex_holt_winters/`: Modul dedicat metodelor de netezire exponențială.
- `arma_analysis/`: Modul pentru analiza și modelarea pur ARMA (AutoRegressive Moving Average).
- `arima_analysis/`: Modul pentru modelarea seriilor integrate (ARIMA) și rezolvarea problemelor de nestaționaritate.

---

## 📈 1. Modulul de Netezire Exponențială (Holt-Winters)
Acest modul utilizează tehnici de nivelare pentru a capta tendințele și sezonalitatea din date.

### Metodologie:
- **Simple Exponential Smoothing (SES)**: Pentru date fără trend evident.
- **Holt's Linear Trend**: Captează trendurile liniare în creștere/scădere.
- **Holt-Winters (Multiplicativ/Aditiv)**: Adaugă componenta sezonieră (ex. 12 luni).

### Vizualizări Holt-Winters:
**Împărțirea datelor (Train/Test):**
![Train vs Test](alex_holt_winters/impartire_train_test.png)

**Compararea metodelor de netezire:**
![Comparatie Netezire](alex_holt_winters/comparatie_netezire_exponentiala.png)

**Prognoza Finală Holt-Winters:**
![Prognoza Holt Winters](alex_holt_winters/grafic_holt_winters_prognoza.png)

---

## 📊 2. Modulul ARMA (Analiză pentru Serii Staționare)
Acest modul implementează un model **ARMA(p, q)** pur (fără diferențiere, d=0).

### Metodologie:
1. **Identificarea Ordinelor (ACF/PACF)**: Ne uităm la funcțiile de autocorelație pentru a deduce `p` (ordinul Autoregresiv) și `q` (ordinul Moving Average).
2. **Grid Search (AIC)**: S-a testat iterativ pentru a minimiza criteriul informațional Akaike, rezultând modelul optim **ARMA(3, 3)**.
3. **Diagnostic**: Evaluarea reziduurilor pentru a confirma că sunt *White Noise* (Zgomot Alb) - folosind testul Ljung-Box.

### Vizualizări ARMA:
**Identificare (Autocorelație ACF și PACF):**
![Identificare ACF PACF](arma_analysis/rezultate/acf_pacf_identificare.png)

**Diagnosticul Modelului (Verificare Zgomot Alb):**
Reziduurile standardizate nu prezintă structură, distribuția este aproximativ normală, iar testul Ljung-Box confirmă validitatea (p-value > 0.05).
![Diagnostic ARMA](arma_analysis/rezultate/diagnostic_model_arma.png)

**Prognoza Finală ARMA:**
Cu interval de încredere de 95%. Lățimea intervalului crește cu orizontul, tinzând către varianța necondiționată.
![Prognoza ARMA](arma_analysis/rezultate/prognoza_finala_arma.png)

---

## 📉 3. Modulul ARIMA (Pentru Serii Nestaționare)
Multe serii macroeconomice sunt nestaționare (prezintă trend stochastic - Mers Aleatoriu). Modulul ARIMA aplică metodologia completă Box-Jenkins.

### Metodologie:
1. **Testul Augmented Dickey-Fuller (ADF)**: A demonstrat că seria nivelurilor este nestaționară, dar devine staționară prin aplicarea primei diferențe (`d=1`).
2. **Selecția Modelului**: Grid Search pentru `p` și `q` pe seria staționarizată a indicat ca optim modelul **ARIMA(2, 1, 3)**.
3. **Validare**: Evaluarea normalității reziduurilor și a autocorelației acestora.

### Vizualizări ARIMA:
**Nivel Original vs. Prima Diferență:**
Se observă cum trendul stochastic dispare și varianța se stabilizează în jurul mediei 0.
![Nivele vs Diferente](arima_analysis/rezultate/nivele_vs_diferenta.png)

**ACF și PACF pe Seria Diferențiată:**
Utile pentru a stabili ordinele componentelor AR și MA.
![ACF PACF ARIMA](arima_analysis/rezultate/acf_pacf_arima.png)

**Diagnosticul Modelului ARIMA:**
Reziduurile sunt stabile, distribuția arată bine, iar ACF-ul reziduurilor nu mai are "spike"-uri semnificative.
![Diagnostic ARIMA](arima_analysis/rezultate/diagnostic_arima.png)

**Prognoza Finală ARIMA:**
Se observă cum incertitudinea crește liniar cu timpul, specific proceselor integrate I(1).
![Prognoza ARIMA Finală](arima_analysis/rezultate/prognoza_arima_finala.png)

---

## 🎯 Concluzii și Performanțe
- Modelele ARIMA și ARMA produc erori similare pe setul de test (MAE ~0.30%).
- Metodologia Box-Jenkins s-a dovedit a fi robustă, iar ambele abordări (staționare/nestaționare) au fost riguros documentate, testele ADF și Ljung-Box confirmând validitatea metodologiei aplicate.
