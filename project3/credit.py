import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score

# Import Data
df = pd.DataFrame.from_csv('credit_dollars.csv')
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)



# GridSearchCV For Each Model
tree = DecisionTreeClassifier()
param_tree = {'max_depth':[10,15,20,None], 'max_features':[10,15,20,None]}

tree_grid = GridSearchCV(tree, param_tree, cv=5, scoring='f1')

tree_grid.fit(X,y)
tree_ypred = grid_tree.predict(X_test)

print('best score: ',tree_grid.best_score_)
print(tree_grid.best_params_)

print('precision:',precision_score(y_test, tree_ypred),'/n','recall:',recall_score(y_test, tree_ypred),'\n','accuracy:',accuracy_score(y_test,tree_ypred))

feat_import = tree_grid.feature_importances_
print('features:',feat_import,'/n','max_index:',feat_import.index(max(feat_import)))


tree_yproba = tree_grid.predict_proba(X_test)[:,1]
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_yproba)

cnf_tree = confusion_matrix(y_test, ypred_tree, labels=None)
plt.imshow(cnf_tree, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.tight_layout()
tick_marks = np.arange(len(tree_grid.classes_))
plt.yticks(tick_marks,tree_grid.classes_,rotation=0)
plt.xticks(tick_marks,tree_grid.classes_,rotation=0)
print(cnf_tree)
plt.savefig('tree_cnf.png')
