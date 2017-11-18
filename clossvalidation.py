import numpy as np
import pandas as pd
import preprocessingTitanic as pt
import classifierSVC as csvc
import classifierRandomForest as crf
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score

data_train = pd.read_csv('input/train.csv')
data_test = pd.read_csv('input/test.csv')

data_train, data_test = pt.preprocessingTitanic(data_train, data_test)
X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

clf = csvc.classifySVC(X_all, y_all)
# clf = crf.classifyRandomForest(X_all, y_all)

def run_kfold(clf):
    kf = KFold(n_splits=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X_all):
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))

run_kfold(clf)
