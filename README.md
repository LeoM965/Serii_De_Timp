# Serii De Timp (Time Series Analysis)

Acest proiect conține seturi de date și instrumente pentru analiza seriilor de timp, concentrându-se pe date demografice și statistice (precum rata șomajului BIM și IPC).

## Structura Proiectului

- `documents/`: Conține seturile de date inițiale în format CSV.
  - `exportPivot_AMG157G.csv`: Date statistice pe grupe de vârstă și sexe.
  - `exportPivot_IPC102A.csv`: Date adiționale pentru analiză.

- `arima_analysis/`: Modul pentru modelare **ARIMA (Integrated)**, destinat seriilor de timp nestaționare (ca PIB sau Șomaj).
  - `main_arima.py`: Script care automatizează testarea ADF, diferențierea (d=1) și selecția p, q.
  - `rezultate/`: Include comparații Nivel vs. Diferență, ACF/PACF pe seria staționară și diagnoza modelului.

## Metode de Analiză
Proiectul utilizează biblioteca `statsmodels` pentru a implementa:
- **Netezire Exponențială**: SES, Holt, Holt-Winters.
- **Modele ARMA**: Pentru serii staționare (p, 0, q).
- **Modele ARIMA**: Pentru serii nestaționare (p, d, q), folosind testul ADF pentru determinarea ordinului de integrare.

## Rezultate și Vizualizări
### Modul ARMA
Model optim identificat: **ARMA(3, 3)** (AIC=94.39). Rădăcinile sunt supraunitare, iar reziduurile sunt zgomot alb.

### Modul ARIMA
Pentru rata șomajului BIM (serie de tip I(1)), modelul optim identificat este **ARIMA(2, 1, 3)** (AIC=84.71).
- **Diferențiere**: Seria a devenit staționară după prima diferențiere (p-value ADF < 0.0001).
- **Validitate**: Reziduurile nu prezintă autocorelație semnificativă (Ljung-Box p = 0.96).

![Prognoză ARIMA](arima_analysis/rezultate/prognoza_arima_finala.png)
