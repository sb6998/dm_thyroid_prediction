  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def xgboost_(feat, tar, split):
    ms = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(feat,tar,test_size=split,random_state=101)
    ms.fit(X_train)
    X_train = ms.transform(X_train)
    X_test = ms.transform(X_test)
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(classification_report(y_test,predictions))