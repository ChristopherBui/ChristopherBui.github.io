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
                          cmap=pylab.cm.Blues):
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


# GridSearchCV For Random Forest
rf = RandomForestClassifier()
rf_param = {}

rf_grid = GridSearchCV(rf, param_grid=rf_param, cv=5, scoring='f1')

rf_grid.fit(X,y)
rf_ypred = rf_grid.predict(X_test)


rf_yproba = rf_grid.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_yproba)

print('best score: ',rf_grid.best_score_)
print(rf_grid.best_params_)

print('precision:',precision_score(y_test, rf_ypred),'\n','recall:',recall_score(y_test, rf_ypred),'\n','accuracy:',accuracy_score(y_test,rf_ypred),'\n','auc:',auc(fpr_rf,tpr_rf))
