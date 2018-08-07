import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
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

# Functions To Be Used
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    pylab.imshow(cm, interpolation='nearest', cmap=cmap)
    pylab.title(title)
    pylab.colorbar()
    tick_marks = np.arange(len(classes))
    pylab.xticks(tick_marks, classes, rotation=45)
    pylab.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pylab.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pylab.tight_layout()
    pylab.ylabel('True label')
    pylab.xlabel('Predicted label')



# GridSearchCV For Each Model
tree = DecisionTreeClassifier()
max_depth = list(range(10,len(X)+1)).append(None)

param_tree = {'max_depth':max_depth, 'max_features':max_depth}

tree_grid = GridSearchCV(tree, param_tree, cv=5, scoring='f1')

tree_grid.fit(X,y)
tree_ypred = tree_grid.predict(X_test)

print('best score: ',tree_grid.best_score_)
print(tree_grid.best_params_)

print('precision:',precision_score(y_test, tree_ypred),'\n','recall:',recall_score(y_test, tree_ypred),'\n','accuracy:',accuracy_score(y_test,tree_ypred))

#feat_import = tree_grid.feature_importances_
#print('features:',feat_import,'/n','max_index:',feat_import.index(max(feat_import)))


tree_yproba = tree_grid.predict_proba(X_test)[:,1]
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_yproba)

cnf_tree = confusion_matrix(y_test, tree_ypred, labels=None)
plot_confusion_matrix(cnf_tree, title='Decision Tree CM', classes=tree_grid.classes_)
pylab.savefig('cnf_tree.png')
