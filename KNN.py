#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('KNN on Wisconsin Brest Cancer Data')


# In[7]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

#column names
col_names = ['Code Number', 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# load dataset
#wbcd = pd.read_csv('wbcd.csv', header=None, names=col_names)
wbcdReplacedData = pd.read_csv('wbcdReplacedData.csv', header=None, names=col_names)
feature_cols = [ 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
features= wbcdReplacedData[feature_cols] # Features
result = wbcdReplacedData.Class # Target variable
featuresReplacedData= wbcdReplacedData[feature_cols] # Features all data
resultReplacedData = wbcdReplacedData.Class # Target variable all data

X_train_, X_test_, y_train_, y_test_ = train_test_split(featuresReplacedData, resultReplacedData, test_size = 0.35)


knn = KNeighborsClassifier(n_neighbors=5
                           
                        , metric='euclidean')
knn.fit(X_train_, y_train_)

y_pred = knn.predict(X_test_)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test_,y_pred))
print(classification_report(y_test_,y_pred))





# In[2]:


# sns.scatterplot(
#     x='mean area',
#     y='mean compactness',
#     hue='benign',
#     data=X_test.join(y_test_, how='outer')
# )

plt.scatter(
    X_test_['Clump Thickness'],
    X_test_['Uniformity of Cell Size'],
    c=y_pred,
    cmap='coolwarm',
    alpha=0.7
)

