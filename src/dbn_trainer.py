from lasagne.layers import *
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from matplotlib import cm
from matplotlib.pyplot import figure, clf, plot, ginput, show, imshow, subplot
import pickle
from sklearn.preprocessing import LabelEncoder

__author__ = 'larisa'

import numpy as np
from time import time
from datetime import timedelta
# import nolearn.lasagne
from nolearn.dbn import DBN
from sklearn import datasets
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
import os


class MyDBN(DBN):
    """ Extension of original fit. For possibility of early stopping when is being learned,
    added accuracy monitoring on held out validation data.
    """
    def fit_with_validation(self, X, y, X_val, y_val, X_test, y_test):
        if self.verbose:
            print "[DBN] fitting X.shape=%s" % (X.shape,)
        self._enc = LabelEncoder()
        y = self._enc.fit_transform(y)
        y = self._onehot(y)

        self.net_ = self._build_net(X, y)

        minibatches_per_epoch = self.minibatches_per_epoch
        if minibatches_per_epoch is None:
            minibatches_per_epoch = X.shape[0] / self.minibatch_size

        loss_funct = self.loss_funct
        if loss_funct is None:
            loss_funct = self._num_mistakes

        errors_pretrain = self.errors_pretrain_ = []
        losses_fine_tune = self.losses_fine_tune_ = []
        errors_fine_tune = self.errors_fine_tune_ = []

        if self.epochs_pretrain:
            self.epochs_pretrain = self._vp(self.epochs_pretrain)
            self._configure_net_pretrain(self.net_)
            for layer_index in range(len(self.layer_sizes) - 1):
                errors_pretrain.append([])
                if self.verbose:  # pragma: no cover
                    print "[DBN] Pre-train layer {}...".format(layer_index + 1)
                time0 = time()
                for epoch, err in enumerate(
                    self.net_.preTrainIth(
                        layer_index,
                        self._minibatches(X),
                        self.epochs_pretrain[layer_index],
                        minibatches_per_epoch,
                        )):
                    errors_pretrain[-1].append(err)
                    if self.verbose:  # pragma: no cover
                        print "  Epoch {}: err {}".format(epoch + 1, err)
                        elapsed = str(timedelta(seconds=time() - time0))
                        print "  ({})".format(elapsed.split('.')[0])
                        time0 = time()
                    if self.pretrain_callback is not None:
                        self.pretrain_callback(
                            self, epoch + 1, layer_index)

        self._configure_net_finetune(self.net_)
        if self.verbose:  # pragma: no cover
            print "[DBN] Fine-tune..."
        time0 = time()
        acc_log = []
        for epoch, (loss, err) in enumerate(
            self.net_.fineTune(
                self._minibatches(X, y),
                self.epochs,
                minibatches_per_epoch,
                loss_funct,
                self.verbose,
                self.use_dropout,
                )):
            losses_fine_tune.append(loss)
            errors_fine_tune.append(err)
            self._learn_rate_adjust()

            # performing validation
            acc = self.score(X_val, y_val)
            # test_acc = self.score(X_test, y_test)
            acc_log.append(acc)
            figure("validation accuracy"); clf()
            plot(np.array(acc_log), "r")
            ginput(1, 0.01)

            if self.verbose:  # pragma: no cover
                print "Epoch {}:".format(epoch + 1)
                print "  loss {}".format(loss)
                print "  err  {}".format(err)
                elapsed = str(timedelta(seconds=time() - time0))
                print "  ({})".format(elapsed.split('.')[0])
                print "  CV accuracy: {}".format(acc)
                # print "  Test accuracy: {}".format(test_acc)
                time0 = time()
            if self.fine_tune_callback is not None:
                self.fine_tune_callback(self, epoch + 1)

        # if self.verbose:
        #     show()


# class DBN_Model:
def train(alphabet, feat_labs, context_size=10, file_name='results.txt',
          n_hidden=3, hidden_size=1024, n_epochs=40, n_epochs_pretrain=0, validation_on=False):
    X = []
    Y = []
    input_size = 13 * (2*context_size+1)
    output_size = len(alphabet)

    # feat_labs.
    # feat_labs = feat_labs[:-97]
    # print len(feat_labs)

    for (xs, ys) in feat_labs:
        for i in xrange(context_size, len(xs) - context_size):
            X.append(np.ndarray(shape=(input_size,), dtype=float, buffer=xs[i-context_size : i+1+context_size]))
            Y.append(ys[i])

    # X = np.array(X).astype(np.float32)
    # Y = np.array(Y).astype(np.int32)
    X = np.array(X)
    Y = np.array(Y)

    print X.shape

    # splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

    # # layers initializing
    # network = InputLayer((None, input_size))
    # for i in xrange(n_hidden):
    #     network = DenseLayer(network, hidden_size)
    # network = DenseLayer(network, output_size, nonlinearity=softmax)
    #
    # model = nolearn.lasagne.NeuralNet(network, max_epochs=40, update=adam, verbose=1)


    layers = [input_size]
    layers.extend(np.repeat(hidden_size, n_hidden))
    layers.append(output_size)
    model = MyDBN(layers,
                learn_rates=0.1,
                learn_rate_decays=0.9,
                epochs=n_epochs,
                epochs_pretrain=n_epochs_pretrain,
                # learn_rates_pretrain=0.01,
                # minibatch_size=128,
                verbose=1, )

    if validation_on:
        # creating held out validation part in train data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
        model.fit_with_validation(X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print "Test accuracy: " + str(acc)

    # writing results in file
    string = "%d utterances, %d context, DBN%s, %d pretrain_epochs, %d epochs, validation %r: accuracy=%f\n" \
             % (len(feat_labs), context_size, model.layer_sizes, model.epochs_pretrain, model.epochs, validation_on, acc)

    newpath = r'../DBN'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    with open('../DBN/'+file_name, 'a') as file:
        file.write(string)

    return model


def predict(model, (xs, ys), context_size=None):
    X = []
    y = []
    input_size = model.get_params()['layer_sizes'][0]
    output_size = model.get_params()['layer_sizes'][-1]
    if context_size is None:
        context_size = (input_size/13 - 1) / 2

    for i in xrange(context_size, len(xs) - context_size):
        X.append(np.ndarray(shape=(input_size,), dtype=float, buffer=xs[i-context_size : i+1+context_size]))
        y.append(ys[i])

    X = np.array(X)
    y = np.array(y)

    def make_targets(classes):
        res = []
        for c in classes:
            array = np.zeros((output_size,))
            array[c-1] = 1
            res.append(array)
        return res

    def most_prob(array):
        res = np.zeros((len(array),))
        res[np.argmax(array)] = 1
        return res

    correct = np.array(make_targets(y))

    pred = model.predict_proba(X)
    # pred = np.array([most_prob(ar) for ar in pred])

    mistakes = model._num_mistakes(pred, correct)
    accuracy = str(1.0 - float(mistakes) / len(y))
    print "score: " + accuracy
    # print model.score(X, y)

    figure("prediction"); clf()
    subplot(211); imshow(pred.T, cmap=cm.hot, interpolation='nearest')
    subplot(212); imshow(correct.T, cmap=cm.hot, interpolation='nearest')
    ginput(1, 11110.01)

    # show()
    return accuracy


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
