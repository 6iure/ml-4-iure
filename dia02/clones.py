# %%

import pandas as pd

df = pd.read_parquet("../data/dados_clones.parquet")
df

# %%
#* Como descobrir onde está o problema?
#* <Estatistica descritiva>

df.groupby(['Status '])[['Estatura(cm)', 'Massa(em kilos)']].mean()

# %%

df['Status_bool'] = df['Status '] == 'Apto'
df

# %%
df.groupby(["Tamanho dos pés"])['Status_bool'].mean()

# %%
df.groupby(["Distância Ombro a ombro"])['Status_bool'].mean()

# %%
df.groupby(["Tamanho do crânio"])['Status_bool'].mean()

# %%
#! Discrepância encontrada
df.groupby(["General Jedi encarregado"])['Status_bool'].mean()

# %%

features = ['Estatura(cm)',
            'Massa(em kilos)',
            "Distância Ombro a ombro",
            "Tamanho do crânio",
            "Tamanho dos pés",
]

#* features categoricas, aquelas que sao "Tipo 1, 2, 3... etc"
cat_features = [ "Distância Ombro a ombro",
            "Tamanho do crânio",
            "Tamanho dos pés" ]

x = df[features]
x

# %%
#* Transformacao de categoria para numerico

from feature_engine import encoding

onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(x)
x = onehot.transform(x)
x

# %% 
from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(x, df['Status '])

# %%

tree.plot_tree(arvore,
               class_names=arvore.classes_,
               feature_names=x.columns,
               filled=True)
