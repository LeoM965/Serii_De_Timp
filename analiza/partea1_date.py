import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=== ETAPA 1: PREGATIREA DATELOR ===")
somaj_df = pd.read_csv(os.path.join(SCRIPT_DIR, 'exportPivot_AMG157G.csv'))
ipc_df = pd.read_csv(os.path.join(SCRIPT_DIR, 'exportPivot_IPC102A.csv'))

somaj_df.columns = somaj_df.columns.str.strip()
ipc_df.columns = ipc_df.columns.str.strip()

date_rng = pd.date_range(start='2010-01-01', periods=192, freq='MS')

df_final = pd.DataFrame({
    'Data': date_rng,
    'Somaj': somaj_df['Valoare'].values,
    'Inflatie_IPC': ipc_df['Valoare'].values
})
df_final.set_index('Data', inplace=True)
df_final.to_csv(os.path.join(SCRIPT_DIR, 'date_proiect_curba_phillips.csv'))


print("=== ETAPA 2: GRAFICE PENTRU DOCUMENTUL WORD ===")
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df_final.index, df_final['Somaj'], color='blue', label='Rata Somajului (Bruta, %)')
plt.title('Evolutia macroeconomica: Somaj si Inflatie (2010 - 2025)')
plt.ylabel('Procente (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df_final.index, df_final['Inflatie_IPC'], color='red', label='Indicele Preturilor de Consum (IPC)')
plt.ylabel('Index IPC')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'grafic_evolutie.png'))
# Am comentat plt.show() ca sa nu iti mai blocheze codul pe ecran
# plt.show() 


print("=== ETAPA 3: SALVAREA TESTELOR ADF IN FISIER TEXT ===")
df_final['Somaj_Diff1'] = df_final['Somaj'].diff()
df_final['Inflatie_Diff1'] = df_final['Inflatie_IPC'].diff()

# Deschidem un fisier text pentru a scrie rezultatele in el
with open(os.path.join(SCRIPT_DIR, 'rezultate_ADF.txt'), 'w', encoding='utf-8') as f:
    f.write("REZULTATELE TESTULUI DE STATIONARITATE (ADF)\n")
    f.write("=============================================\n\n")

    def testeaza_si_scrie(serie, nume_serie):
        f.write(f"--- Testul ADF pentru: {nume_serie} ---\n")
        rezultat = adfuller(serie.dropna()) 
        f.write(f"Statistica ADF: {rezultat[0]:.4f}\n")
        f.write(f"P-value: {rezultat[1]:.4f}\n")
        
        if rezultat[1] <= 0.05:
            f.write("Concluzie: Respingem ipoteza nula. Seria ESTE STATIONARA (I(0)).\n\n")
        else:
            f.write("Concluzie: Nu putem respinge ipoteza nula. Seria are un TREND STOCHASTIC (Non-Stationara).\n\n")

    # Scriem rezultatele pentru seriile la nivel
    testeaza_si_scrie(df_final['Somaj'], "Somaj (La Nivel)")
    testeaza_si_scrie(df_final['Inflatie_IPC'], "Inflatie IPC (La Nivel)")
    
    f.write(">>> Aplicam diferentierea de ordinul 1... <<<\n\n")
    
    # Scriem rezultatele pentru seriile diferentiate
    testeaza_si_scrie(df_final['Somaj_Diff1'], "Somaj (Diferentiat - Ordinul 1)")
    testeaza_si_scrie(df_final['Inflatie_Diff1'], "Inflatie IPC (Diferentiat - Ordinul 1)")

print("Gata! Verifica folderul, a aparut fisierul 'rezultate_ADF.txt'.")

# CURBA PHILLIPS
# 1. Incarcam datele pe care le-ai curatat la pasul anterior
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'date_proiect_curba_phillips.csv'))

# 2. Cream graficul de tip Scatter Plot (Curba Phillips)
plt.figure(figsize=(9, 6))

# seaborn face automat si punctele, si linia de trend
sns.regplot(x='Somaj', y='Inflatie_IPC', data=df, 
            scatter_kws={'alpha':0.6, 'color':'blue'}, 
            line_kws={'color':'red', 'linewidth':2})

plt.title('Curba Phillips Empirica in Romania (2010 - 2025)')
plt.xlabel('Rata Somajului (%)')
plt.ylabel('Indicele Preturilor de Consum (IPC)')
plt.grid(True, linestyle='--', alpha=0.5)

# 3. Salvam graficul pentru a-l pune in Word
plt.savefig(os.path.join(SCRIPT_DIR, 'curba_phillips_scatter.png'))
print("Graficul a fost salvat ca 'curba_phillips_scatter.png'.")