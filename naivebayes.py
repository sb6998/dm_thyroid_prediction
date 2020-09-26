

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def naive_bayes(feat, tar, split):
  ms = MinMaxScaler()
  X_train, X_test, y_train, y_test = train_test_split(feat,tar,test_size=split,random_state=101)
  ms.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)
  nb = MultinomialNB()
  nb.fit(X_train,y_train)
  predictions = nb.predict(X_test)
  print(classification_report(y_test,predictions))
