from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def test_tree():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert np.allclose(accuracy_score(y_test, preds), 0.973684210526)
