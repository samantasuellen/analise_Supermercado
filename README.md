# analise_Supermercado
Este projeto acadêmico tem como objetivo aplicar conceitos básicos de Estatística Aplicada e Machine Learning introdutório utilizando dados fictícios de um supermercado. A análise foi desenvolvida para compreender o comportamento das vendas, custos e faturamento, auxiliando a tomada de decisões de forma simples e prática.

### Exemplo

```
# Projeto Acadêmico - Análise de Dados de Supermercado
# Estatística Aplicada + Machine Learning Básico
# Dados fictícios

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1. Criando dados fictícios
# -----------------------------
np.random.seed(1)

meses = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez"
]

volume_vendas = np.random.randint(800, 1500, 12)
preco_medio = np.random.uniform(8, 15, 12)
custos_fixos = np.random.uniform(4000, 6000, 12)
custos_variaveis = volume_vendas * np.random.uniform(2, 4, 12)

faturamento = volume_vendas * preco_medio
custos_totais = custos_fixos + custos_variaveis
lucro = faturamento - custos_totais

df = pd.DataFrame({
    "Mes": meses,
    "Volume_Vendas": volume_vendas,
    "Faturamento": faturamento,
    "Custos_Totais": custos_totais,
    "Lucro": lucro
})

print("Dados do Supermercado:")
print(df)

# -----------------------------
# 2. Estatística Descritiva
# -----------------------------
media_vendas = df["Volume_Vendas"].mean()
desvio_padrao = df["Volume_Vendas"].std()

# Margem de erro (95%)
n = len(df)
margem_erro = 1.96 * (desvio_padrao / np.sqrt(n))

print("\nEstatísticas:")
print(f"Média de Vendas: {media_vendas:.2f}")
print(f"Margem de Erro (95%): ±{margem_erro:.2f}")

# -----------------------------
# 3. Gráficos
# -----------------------------
plt.figure()
plt.plot(df["Mes"], df["Volume_Vendas"])
plt.title("Volume de Vendas Mensal")
plt.xlabel("Mês")
plt.ylabel("Unidades Vendidas")
plt.show()

plt.figure()
plt.plot(df["Mes"], df["Faturamento"], label="Faturamento")
plt.plot(df["Mes"], df["Custos_Totais"], label="Custos")
plt.title("Faturamento x Custos")
plt.xlabel("Mês")
plt.ylabel("Valor (R$)")
plt.legend()
plt.show()

plt.figure()
plt.bar(df["Mes"], df["Lucro"])
plt.title("Lucro Mensal")
plt.xlabel("Mês")
plt.ylabel("Lucro (R$)")
plt.show()

# -----------------------------
# 4. Machine Learning Básico
# -----------------------------
# Prever faturamento com base no volume de vendas

X = df[["Volume_Vendas"]]
y = df["Faturamento"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)

erro = mean_absolute_error(y_teste, previsoes)

print("\nMachine Learning:")
print(f"Erro médio absoluto (MAE): {erro:.2f}")

# Comparação real x previsto
resultado = pd.DataFrame({
    "Faturamento Real": y_teste.values,
    "Faturamento Previsto": previsoes
})

print("\nComparação Real x Previsto:")
print(resultado)

```
### Gráfico


<img width="423" height="312" alt="lucro mensal" src="https://github.com/user-attachments/assets/5f317456-fd32-460e-9149-160ce684442e" />
<img width="423" height="309" alt="vendas mensal" src="https://github.com/user-attachments/assets/bde34eda-54e6-4d9d-9df1-f77629713d3a" />
<img width="424" height="314" alt="Faturamento e custos" src="https://github.com/user-attachments/assets/00969de6-8a6b-46e2-a71c-b4fa61737c19" />


