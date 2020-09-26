from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def nn(feat,tar,split):
    scaler = StandardScaler()
    encoder = LabelEncoder()
    encoder.fit(tar)
    encoded_Y = encoder.transform(tar)
    dummy_y = np_utils.to_categorical(encoded_Y)
    x_tr,x_te,y_tr,y_te = train_test_split(feat,dummy_y,test_size = split,shuffle = True)
    scaler.fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_te = scaler.transform(x_te)
    model = tf.keras.Sequential([
        layers.Dense(64,input_dim=31, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(x_tr,y_tr,epochs=50,batch_size=40,validation_split=0.2)
    loss, accuracy = model.evaluate(x_te,y_te)
    print("Accuracy", accuracy)
    return history