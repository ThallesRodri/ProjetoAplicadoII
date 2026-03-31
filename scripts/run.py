import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "bmw_global_sales_2018_2025.csv"
OUTPUT_DIR = "bmw_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 1) Leitura da base
df = pd.read_csv(DATA_PATH)

# 2) Limpeza e preparação
df.columns = [col.strip() for col in df.columns]
df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))

# Verificações básicas
missing_values = df.isnull().sum()
duplicated_rows = df.duplicated().sum()

# Regra de consistência da receita
df["Revenue_Check"] = df["Units_Sold"] * df["Avg_Price_EUR"]
df["Revenue_Consistent"] = df["Revenue_Check"] == df["Revenue_EUR"]

# 3) Análise exploratória
annual = df.groupby("Year", as_index=False).agg(
    Units_Sold=("Units_Sold", "sum"),
    Revenue_EUR=("Revenue_EUR", "sum"),
    BEV_Share=("BEV_Share", "mean")
)

regional = df.groupby("Region", as_index=False).agg(
    Units_Sold=("Units_Sold", "sum"),
    Revenue_EUR=("Revenue_EUR", "sum")
).sort_values("Units_Sold", ascending=False)

correlation = df[
    ["Units_Sold", "Avg_Price_EUR", "Revenue_EUR", "BEV_Share",
     "Premium_Share", "GDP_Growth", "Fuel_Price_Index"]
].corr()

# 4) Agregação mensal para modelagem preditiva
monthly = df.groupby("Date", as_index=False).agg(
    Units_Sold=("Units_Sold", "sum"),
    Avg_Price_EUR=("Avg_Price_EUR", "mean"),
    Revenue_EUR=("Revenue_EUR", "sum"),
    BEV_Share=("BEV_Share", "mean"),
    Premium_Share=("Premium_Share", "mean"),
    GDP_Growth=("GDP_Growth", "mean"),
    Fuel_Price_Index=("Fuel_Price_Index", "mean")
)

monthly["t"] = np.arange(len(monthly))
monthly["month_num"] = monthly["Date"].dt.month
monthly["sin_month"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
monthly["cos_month"] = np.cos(2 * np.pi * monthly["month_num"] / 12)

features = [
    "t", "sin_month", "cos_month", "Avg_Price_EUR",
    "BEV_Share", "Premium_Share", "GDP_Growth", "Fuel_Price_Index"
]
target = "Units_Sold"

X = monthly[features]
y = monthly[target]

split_idx = int(len(monthly) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

linreg = LinearRegression()
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    min_samples_leaf=2
)

linreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

pred_lin = linreg.predict(X_test)
pred_rf = rf.predict(X_test)

metrics = pd.DataFrame([
    {
        "Modelo": "Regressão Linear",
        "MAE": mean_absolute_error(y_test, pred_lin),
        "RMSE": mean_squared_error(y_test, pred_lin) ** 0.5,
        "R2": r2_score(y_test, pred_lin),
        "MAPE (%)": mape(y_test, pred_lin),
    },
    {
        "Modelo": "Random Forest",
        "MAE": mean_absolute_error(y_test, pred_rf),
        "RMSE": mean_squared_error(y_test, pred_rf) ** 0.5,
        "R2": r2_score(y_test, pred_rf),
        "MAPE (%)": mape(y_test, pred_rf),
    }
])

# 5) Geração de gráficos
plt.figure(figsize=(8, 4.5))
plt.plot(annual["Year"], annual["Units_Sold"] / 1e6, marker="o")
plt.title("Vendas anuais da BMW (milhões de unidades)")
plt.xlabel("Ano")
plt.ylabel("Unidades vendidas (milhões)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_vendas_anuais.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 4.5))
plt.bar(regional["Region"], regional["Units_Sold"] / 1e6)
plt.title("Vendas acumuladas por região (2018-2025)")
plt.xlabel("Região")
plt.ylabel("Unidades vendidas (milhões)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_vendas_regiao.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 4.5))
plt.plot(annual["Year"], annual["BEV_Share"] * 100, marker="o")
plt.title("Evolução média da participação de veículos elétricos (BEV)")
plt.xlabel("Ano")
plt.ylabel("BEV Share (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_bev_share.png"), dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(correlation.values, aspect="auto")
ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.index)))
ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
ax.set_yticklabels(correlation.index)
for i in range(correlation.shape[0]):
    for j in range(correlation.shape[1]):
        ax.text(j, i, f"{correlation.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Matriz de correlação das variáveis numéricas")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_correlacao.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 4.5))
plt.plot(monthly["Date"].iloc[split_idx:], y_test.values, marker="o", label="Real")
plt.plot(monthly["Date"].iloc[split_idx:], pred_rf, marker="o", label="Previsto")
plt.title("Comparação entre vendas reais e previstas")
plt.xlabel("Data")
plt.ylabel("Unidades vendidas")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_real_previsto.png"), dpi=200)
plt.close()

# 6) Exportação de tabelas de apoio
annual.to_csv(os.path.join(OUTPUT_DIR, "resumo_anual.csv"), index=False)
regional.to_csv(os.path.join(OUTPUT_DIR, "resumo_regional.csv"), index=False)
metrics.to_csv(os.path.join(OUTPUT_DIR, "metricas_modelos.csv"), index=False)

print("Valores ausentes por coluna:")
print(missing_values)
print("\nLinhas duplicadas:", duplicated_rows)
print("\nConsistência da receita:", df["Revenue_Consistent"].mean() * 100, "%")
print("\nMétricas dos modelos:")
print(metrics.round(4))
