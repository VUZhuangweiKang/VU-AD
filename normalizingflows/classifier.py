from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score

def RFClassifier(X_train, y_train, X_test=None, y_test=None):
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train)

    score_ = None
    if X_test is not None and y_test is not None:
        predicted = clf.predict(X_test)
        score_ = score(y_test, predicted)
    return clf, score_