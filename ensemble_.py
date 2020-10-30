from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def ensemble_(feat, tar, split):
    scaler = MinMaxScaler()
    x_tr,x_te,y_tr,y_te = train_test_split(feat,tar,test_size = split,shuffle = True)
    scaler.fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_te = scaler.transform(x_te)
    
    knn = KNeighborsClassifier()
    params_knn = {'n_neighbors': np.arange(1, 25)}
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    knn_gs.fit(x_tr, y_tr)
    knn_best = knn_gs.best_estimator_
    print(knn_gs.best_params_)
    
    rf = RandomForestClassifier()
    params_rf = {'n_estimators': [50, 100, 200,300,400]}
    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    rf_gs.fit(x_tr, y_tr)
    rf_best = rf_gs.best_estimator_
    print(rf_gs.best_params_)
    
    
    log_reg = LogisticRegression()
    log_reg.fit(x_tr, y_tr)
    
    print('knn: {}'.format(knn_best.score(x_te, y_te)))
    print('rf: {}'.format(rf_best.score(x_te, y_te)))
    print('log_reg: {}'.format(log_reg.score(x_te, y_te)))
    
    estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(x_tr, y_tr)
    print("ensemble voting score: ",str(ensemble.score(x_te, y_te)))
    
    ensemble_bagging = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=10) 
    ensemble_bagging.fit(x_tr, y_tr)
    print("ensemble bagging score: ",str(ensemble_bagging.score(x_te, y_te)))
    
    ensemble_stacking = StackingClassifier(estimators,LogisticRegression())
    ensemble_stacking.fit(x_tr, y_tr)
    print("ensemble stacking score: ", str(ensemble_stacking.score(x_te, y_te)))