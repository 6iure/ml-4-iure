# %% 
import pandas as pd

df = pd.read_excel('../data/dados_cerveja.xlsx')
df

# %%

features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

x = df[features]
y = df[target]

# %%

x = x.replace({
    'mud':1, 'pint':0,
    'sim':1, 'não':0,
    'escura':1, 'clara':0,
})

x

# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier()
arvore.fit(x,y)

# %%

#* escolehu a pilsen como primeira posi tem maior quantidade perante o total

tree.plot_tree(arvore,
               class_names= arvore.classes_,
               feature_names= features,
               filled=True)
# %%

probas = arvore.predict_proba([[-1, 1, 0, 1]])[0]

pd.Series(probas, index=arvore.classes_)
# %%
