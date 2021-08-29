from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score


def RFClassifier(X_train, y_train, X_test=None, y_test=None):
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train)

    score = None
    if X_test and y_test:
        predicted = clf.predict(X_test)
        score = score(y_test, predicted)
    return clf, score