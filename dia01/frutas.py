# %% 
import pandas as pd

df = pd.read_excel("../data/dados_frutas.xlsx")
df

# %% definindo filtro por filtro para descobrir qual é a fruta 

filtro_redonda = df['Arredondada'] == 1
filtro_suculenta = df['Suculenta'] == 1
filtro_vermelha = df['Vermelha'] == 1
filtro_doce = df['Doce'] == 1

df[filtro_redonda & filtro_suculenta & filtro_vermelha & filtro_doce]

# %% Como fazer para a maquina aprender isso?

from sklearn import tree

features = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
target = "Fruta"

x = df[features]
y = df[target]

# %%

arvore = tree.DecisionTreeClassifier()
arvore.fit(x, y)

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=600)

tree.plot_tree(arvore, 
               class_names=arvore.classes_, 
               feature_names=features,
               filled=True)
# %%
# 'Arredondada', 'Suculenta', 'Vermelha', 'Doce'
#* como o predict só retorna 1 resultado, se tivermos empate, retornará em ordem alfabetica

arvore.predict([[0, 1, 1, 1]])

# %% Devolve uma lista de probabilidade para cada fruta'

probas = arvore.predict_proba([[1, 1, 1, 1]])[0]
pd.Series(probas, index=arvore.classes_)


# %%
