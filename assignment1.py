import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from statistics import mean,stdev
import keras
from keras import layers

def load_dataset(file, p_train=0.5):
    df = pd.read_csv(file)
    del df['Time']

    X = np.asarray(df.drop(['Class'],axis=1))
    y = np.asarray(df['Class'])

    del df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_train, shuffle=False)

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, hparams={}):

    #initialize dictionary
    if not(hparams.__contains__('learning_rate')):
        hparams['learning_rate'] = 0.01
    if not(hparams.__contains__('epochs')):
        hparams['epochs'] = 10

    m_train = []
    s_train = []
    for i in range(X_train.shape[1]):
        x = []
        for j in range(X_train.shape[0]):
            x.append(X_train[j][i])
        m_train.append(mean(x))
        s_train.append(stdev(x))

    for i in range(X_train.shape[1]):
        for j in range(X_train.shape[0]):
            X_train[j][i] = (X_train[j][i]-m_train[i])/s_train[i]

    model = keras.Sequential()
    initializer = keras.initializers.RandomNormal(mean=0., stddev=1., seed=12345)
    model.add(layers.Dense(20,activation="relu", kernel_initializer=initializer))
    model.add(layers.Dense(60,activation="relu"))
    model.add(layers.Dense(20,activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))

    opt = keras.optimizers.SGD(learning_rate=hparams['learning_rate'])
    model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])
    model.fit(X_train,y_train,epochs=hparams['epochs'],shuffle=False)

    return model, m_train, s_train

def evaluate_model(model, X_test, y_test, m, s):

    for i in range(X_test.shape[1]):
        for j in range(X_test.shape[0]):
            X_test[j][i] = (X_test[j][i]-m[i])/s[i]

    pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test,pred)

    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)

    return pred, accuracy, precision, recall, f1