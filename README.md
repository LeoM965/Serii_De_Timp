# Serii De Timp (Time Series Analysis)

Acest proiect conține seturi de date și instrumente pentru analiza seriilor de timp, concentrându-se pe date demografice și statistice (precum rata șomajului BIM și IPC).

## Structura Proiectului

- `documents/`: Conține seturile de date inițiale în format CSV.
  - `exportPivot_AMG157G.csv`: Date statistice pe grupe de vârstă și sexe.
  - `exportPivot_IPC102A.csv`: Date adiționale pentru analiză.

- `alex_holt_winters/`: Modul de analiză și prognoză folosind metode de netezire exponențială.
  - `main.py`: Scriptul principal pentru procesarea datelor, antrenarea modelelor și generarea prognozelor.
  - `date/`: Seturi de date specifice pentru modelele Holt-Winters (`somaj_bim.csv`, `ipc_inflatie.csv`).
  - `rezultate/`: Include grafice (`.png`) și tabele de rezultate (`.csv`) cu erorile de prognoză (MAE, RMSE, MAPE).

## Metode de Analiză
Proiectul utilizează biblioteca `statsmodels` pentru a implementa:
- Simple Exponential Smoothing (SES)
- Holt's Linear Trend Method
- Holt-Winters Seasonal Method (Aditiv și Multiplicativ)

## Rezultate și Vizualizări
Modelele sunt evaluate pe un set de date de test (2024-2025), generând comparații vizuale între valorile reale și cele prognozate.
