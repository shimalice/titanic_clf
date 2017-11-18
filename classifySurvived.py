import pandas as pd
import preprocessingTitanic as pt
import classifierRandomForest as crf

data_train = pd.read_csv('input/train.csv')
data_test = pd.read_csv('input/test.csv')

data_train, data_test = pt.preprocessingTitanic(data_train, data_test)
X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

clf = crf.classifyRandomForest(X_all, y_all)

ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)
