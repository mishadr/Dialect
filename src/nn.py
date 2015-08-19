__author__ = 'larisa'

import numpy as np
from nolearn.dbn import DBN
from sklearn import datasets
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

def train(X, Y, alphabet):
    model = DBN([13, 1000, len(alphabet)],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=1,)

    model.fit(X, Y)
    return model

def predict(model, feature):
    return model.predict(feature)

def cross_valid(X, Y, alphabet):
    model = train(X, Y, alphabet)
    scores = cross_validation.cross_val_score(model, X, Y, cv=2)
    accuracy = scores.mean(), scores.std() / 2
    return accuracy

def oneholdout(X, Y, alphabet) :
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    model = train(X_train, y_train, alphabet)
    acc = model.score(X_test, y_test)
    print(acc)
