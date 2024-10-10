# %%

import pandas as pd
from sklearn import model_selection

df = pd.read_csv('../data/dados_pontos.csv', sep=';')
df

# %%
#* o primeiro a se fazer é dividir a base em base de treino e base de teste

features = df.columns[3:-1]
target = 'flActive'

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                   df[target],
                                                                   test_size=0.2,
                                                                   random_state=42,
                                                                   stratify=df[target])
print('tx resposta treino:', y_train.mean())
print('tx resposta teste:', y_test.mean())

# %%

input_avgRecorrencia = X_train['avgRecorrencia'].max()

X_train['avgRecorrencia'] = X_train['avgRecorrencia'].fillna(input_avgRecorrencia)

X_test['avgRecorrencia'] = X_test['avgRecorrencia'].fillna(input_avgRecorrencia)

# %%

from sklearn import tree
from sklearn import metrics

#*Aqui treinamos o algoritmo
arvore = tree.DecisionTreeClassifier(max_depth=10,
                                     min_samples_leaf=50,
                                     random_state=42)
arvore.fit(X_train, y_train)

#*Aqui prevemos na propria base 
tree_pred_train = arvore.predict(X_train)
tree_acc_train = metrics.accuracy_score(y_train, tree_pred_train)
print('Arvore train acc:', tree_acc_train)

#*Aqui prevemos na base de teste 
tree_pred_test = arvore.predict(X_test)
tree_acc_test = metrics.accuracy_score(y_test, tree_pred_test)
print('Arvore test acc:', tree_acc_test)

#*Aqui prevemos na propria base 
tree_proba_train = arvore.predict_proba(X_train)[:,1]
tree_acc_train = metrics.roc_auc_score(y_train, tree_proba_train)
print('Arvore train auc:', tree_acc_train)

#*Aqui prevemos na base de teste 
tree_proba_test = arvore.predict_proba(X_test)[:,1]
tree_acc_test = metrics.roc_auc_score(y_test, tree_proba_test)
print('Arvore test auc:', tree_acc_test)

# %%

#*Modelo onde todos sao churn (assistiram a live uma vez e nao voltaram mais)

y_predict_teo = [0 for i in y_test]
acc_teo = metrics.accuracy_score(y_test, y_predict_teo)
acc_teo
