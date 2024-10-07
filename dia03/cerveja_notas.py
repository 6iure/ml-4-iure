# %% 

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('../data/dados_cerveja_nota.xlsx')
df

# %%

plt.plot(df['cerveja'], df['nota'], 'o')
plt.grid(True)
plt.title('Relação Nota vs Cerveja')
plt.xlabel('cerveja')
plt.ylabel('nota')

# %% 
#* como trazer a reta de regressão linear

from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(df[['cerveja']], df['nota'])

# %%

a, b = reg.intercept_, reg.coef_[0]
print(f"a={a}: b={b}")

# %%

x = df[['cerveja']].drop_duplicates()
y_estimado = reg.predict(x)
y_estimado

plt.plot(df['cerveja'], df['nota'], 'o')
plt.plot(x, y_estimado, '-')
plt.grid(True)
plt.title('Relação Nota vs Cerveja')
plt.xlabel('cerveja')
plt.ylabel('nota')

# %%
#todo reg_linear é a linha laranja
#* arvore é a linha verde

from sklearn import tree

arvore = tree.DecisionTreeRegressor(max_depth=2)
arvore.fit(df[['cerveja']], df['nota'])

y_estimado_arvore = arvore.predict(x)

plt.plot(df['cerveja'], df['nota'], 'o')
plt.plot(x, y_estimado, '-')
plt.plot(x, y_estimado_arvore, '-')
plt.grid(True)
plt.title('Relação Nota vs Cerveja')
plt.xlabel('cerveja')
plt.legend(['pontos', 'regressão', 'arvore'])
plt.ylabel('nota')
