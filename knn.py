from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def knn_(feat, tar, split):
    scaler = StandardScaler()
    x_tr,x_te,y_tr,y_te = train_test_split(feat,tar,test_size = split,shuffle = True)
    scaler.fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_te = scaler.transform(x_te)
    k_range = range(2,26)
    scores ={}
    s_list = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_tr, y_tr)
        y_pred = knn.predict(x_te)
        scores[k]=metrics.accuracy_score(y_te,y_pred)
        s_list.append(scores[k])
        #print(confusion_matrix(y_te, y_pred))
        print(k)
        print(classification_report(y_te, y_pred))
