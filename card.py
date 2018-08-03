
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import roc_curve, auc


# In[2]:

df = pd.DataFrame.from_csv('credit_dollars.csv')
print(df.columns)


# In[3]:

df.shape


# In[4]:

df.head()


# In[5]:

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

print(X.shape, y.shape)


# In[6]:

y.value_counts()


# In[7]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)


# In[8]:

#model_svm = svm.SVC(kernel='linear')
#model_svm.fit(X_train, y_train) 
#y_pred = model_svm.predict(X_test)


# In[104]:

# decision tree cross validation
tree = DecisionTreeClassifier()
tree_score = cross_val_score(tree, X, y, cv=10, scoring='accuracy')

avg_tree_score = np.mean(tree_score)
print('(decision tree) avg acc: ', avg_tree_score)


# In[119]:

# ROC Curve For One Decision Tree
tree.fit(X_train, y_train)
tree_proba = tree.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, tree_proba)
tree_auc = auc(fpr, tpr)
print('(decision tree) auc score: ', tree_auc)

plt.plot(fpr, tpr, color='#f20068')
plt.xlabel('FPR')
plt.ylabel('TPR', rotation=0, labelpad=30)
plt.title('Decision Tree ROC')
plt.plot([0,1],[0,1], linestyle='dashed', color = '#171dc4')
plt.show()


# In[9]:

# random forrest
forest = RandomForestClassifier(n_estimators=100, max_features = 15)
forest_score = cross_val_score(forest, X, y, cv=10, scoring='accuracy')


# In[10]:

avg_forest_score = np.mean(forest_score)
print('(random forest) avg acc: ', avg_forest_score)


# In[11]:

# one random forest; max_features = 15
forest.fit(X_train, y_train)
forest_proba = forest.predict_proba(X_test)[:,1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, forest_proba)
random_forest_auc = auc(fpr_rf, tpr_rf)

print('(random forest) auc score: ', random_forest_auc)

plt.plot(fpr_rf, tpr_rf, color='#f20068')
plt.plot([0,1],[0,1], linestyle='dashed', color='#171dc4')
plt.xlabel('FPR')
plt.ylabel('TPR', rotation=0, labelpad=30)
plt.title('Random Forest ROC; 15/23 features')
plt.show()


# In[111]:

# Gradient Boosting
grad = GradientBoostingClassifier()
grad_score = cross_val_score(grad, X, y, cv=10, scoring='accuracy')


# In[112]:

avg_grad_score = np.mean(grad_score)
print('(gradient boosting) avg acc: ', avg_grad_score)


# In[122]:

# one gradient boost
grad.fit(X_train, y_train)
grad_proba = grad.predict_proba(X_test)[:,1]

fpr_gb, tpr_gb, _ = roc_curve(y_test, grad_proba)
gradient_boost_auc = auc(fpr_gb, tpr_gb)

print('(gradient boost) auc score: ', gradient_boost_auc)

plt.plot(fpr_gb, tpr_gb, color='#f20068')
plt.plot([0,1],[0,1], linestyle='dashed', color='#171dc4')
plt.xlabel('FPR')
plt.ylabel('TPR', rotation=0, labelpad=30)
plt.title('Gradient Boost ROC')
plt.show()


# In[ ]:



