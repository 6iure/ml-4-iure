# %%

import pandas as pd

df = pd.read_excel('../data/dados_cerveja_nota.xlsx')
df

# %%

df['aprovado'] = df['nota'] >= 5
df

# %%
from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None,
                                       fit_intercept=True)

features = ['cerveja']
target = 'aprovado'

#* aqui o modelo aprende
reg.fit(df[features], df[target])

#* aqui o modelo prevê
reg_predict = reg.predict(df[features])
reg_predict

# %%

from sklearn import metrics

#* Acurácia: Separa entre é fraude / não é fraude
reg_acc = metrics.accuracy_score(df[target], reg_predict)
print('acurácia reg log', reg_acc)

#* Precisão: taxa de verdadeiros positivos / falsos positivos + vp
reg_precision = metrics.precision_score(df[target], reg_predict)
print("precisão reg log: ", reg_precision)

#* Recall: Quanto dos positivos meu modelo capturou
reg_recall = metrics.recall_score(df[target], reg_predict)
print('recall reg log:', reg_recall)

#* Especifidade: Falso positivo / Verdadeira Neg. + Falso neg.

#* Matriz de confusão 
reg_conf = metrics.confusion_matrix(df[target], reg_predict)
reg_conf = pd.DataFrame(reg_conf, index=['False', 'True'],
                        columns=['false', 'true'])

print(reg_conf)
# %%
#* testando arvore de decisao

from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=2)

#* aqui o modelo aprende
arvore.fit(df[features], df[target])

#* aqui o modelo prevê
arvore_predict = arvore.predict(df[features])
arvore_predict

arvore_acc = metrics.accuracy_score(df[target], arvore_predict)
print('acurácia arvore :', arvore_acc)

arvore_precision = metrics.precision_score(df[target], arvore_predict)
print("precisão arvore: ", arvore_precision)

arvore_recall = metrics.recall_score(df[target], arvore_predict)
print('recall arvore:', arvore_recall)

arvore_conf = metrics.confusion_matrix(df[target], arvore_predict)
arvore_conf

# %% 
#* testando naive bayes

from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()

#* aqui o modelo aprende
nb.fit(df[features], df[target])

#* aqui o modelo prevê
nb_predict = nb.predict(df[features])
nb_predict

nb_acc = metrics.accuracy_score(df[target], nb_predict)
print('acurácia nb:', nb_acc)

nb_precision = metrics.precision_score(df[target], nb_predict)
print("precisão nb:", nb_precision)

nb_recall = metrics.recall_score(df[target], nb_predict)
print('recall nb:', nb_recall)

nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf

# %% 
#* mexendo no ponto de corte da probabilidade, as metricas mudam mas pq 
#* ou seja dpendo do ponto de corte pra escolher o melhor modelo

nb_proba = nb.predict_proba(df[features])[:,1]
nb_predict = nb_proba > 0.2

nb_acc = metrics.accuracy_score(df[target], nb_predict)
print('acurácia nb:', nb_acc)

nb_precision = metrics.precision_score(df[target], nb_predict)
print("precisão nb:", nb_precision)

nb_recall = metrics.recall_score(df[target], nb_predict)
print('recall nb:', nb_recall)

# %%

import matplotlib.pyplot as plt

roc_curve = metrics.roc_curve(df[target], nb_proba)

plt.plot(roc_curve[0], roc_curve[1])
plt.grid(True)
plt.plot([0,1], [0,1], '--')
plt.show()

# %%

roc_auc = metrics.roc_auc_score(df[target], nb_proba)
roc_auc
# %%
