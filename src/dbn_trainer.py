__author__ = 'larisa'

import numpy as np
from nolearn.dbn import DBN
from sklearn import datasets
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
import os


def train(alphabet, feat_labs, context_size=0, file_name='results.txt', hidden_size=512, n_hidden=4):
    X = []
    Y = []
    input_size = 13 * (2*context_size+1)

    # splitting randomly into train and test datasets
    # random_split = cross_validation.ShuffleSplit(len(feat_labs), n_iter=1, test_size=0.10, random_state=0)
    # for (train_index, test_index) in random_split:

    for (xs, ys) in feat_labs:
        for i in xrange(context_size, len(xs) - context_size - 1):
            X.append(np.ndarray(shape=(input_size,), buffer=xs[i-context_size : i+1+context_size]))
            Y.append(ys[i])

    X = np.array(X)
    Y = np.array(Y)

    # splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

    layers = [input_size]
    layers.extend(np.repeat(hidden_size, n_hidden))
    layers.append(len(alphabet))
    model = DBN(layers,
                learn_rates=0.1,
                learn_rate_decays=0.9,
                epochs=12,
                # epochs_pretrain=10,
                # learn_rates_pretrain=0.01,
                # minibatch_size=128,
                verbose=1, )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print "accuracy: " + str(acc)

    # write results in file

    string = "%d utterances, %d context, DBN%s, %d epochs: accuracy=%f\n" %\
             (len(feat_labs), context_size, model.layer_sizes, model.epochs, acc)

    newpath = r'../DBN'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    with open("../DBN/"+file_name, 'a') as file:
        file.write(string)


# def predict(model, feature):
#     return model.predict(feature)
#
#
# def cross_valid(X, Y, alphabet):
#     model = train(X, Y, alphabet)
#     scores = cross_validation.cross_val_score(model, X, Y, cv=2)
#     accuracy = scores.mean(), scores.std() / 2
#     return accuracy


def oneholdout(X, Y, alphabet):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    model = train(X_train, y_train, alphabet)
    acc = model.score(X_test, y_test)
    print "accuracy: " + str(acc)
