import sys
import pandas
import numpy
import matplotlib
import scipy
import seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%%
data = pandas.read_csv('creditcard.csv')
print(data.shape)
data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)

#%%
data.hist(figsize = (20, 20))
plt.show()

#%%
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_frac = (len(fraud)/float(len(data['Class'])))
print('Fraud cases: {}'.format(len(fraud)))

#%%
corrmat = data.corr()

fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

#%%
columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]

target = "Class"

x = data[columns]
y = data[target]

print(x.shape)
print(y.shape)

#%%
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#%%
state = 1

classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(x),
        contamination=outlier_frac,
        random_state=state
    ),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors = 20,
        contamination = outlier_frac
    )
}

#%%
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(x)
        score_pred = clf.negative_outlier_factor_
    else:
         clf.fit(x)
         score_pred = clf.decision_function(x)
         y_pred = clf.predict(x)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_error = (y_pred != y).sum

    print('{}: {}'.format(clf_name, n_error))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))
 

#%%
