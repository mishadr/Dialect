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

    def __init__(self, pretrain_momentum,
        *args, **kwargs
        ):
        super(MyDBN, self).__init__(*args, **kwargs)

        self.pretrain_momentum = pretrain_momentum
        self.fine_tune_momentum = self.momentum

    def fit_with_validation(self, X_pre, X, y, X_val, y_val, output_size):
        if self.verbose:
            print "[DBN] fitting X.shape=%s" % (X.shape,)
        self._enc = LabelEncoder()
        # y = self._enc.fit_transform(y)
        # y = self._onehot(y)

        y = _make_targets(y, output_size)
        self.net_ = self._build_net(X, y)  # doesn't use the data, just its shape

        minibatches_per_epoch = self.minibatches_per_epoch
        if minibatches_per_epoch is None:
            minibatches_per_epoch = X.shape[0] / self.minibatch_size

        loss_funct = self.loss_funct
        if loss_funct is None:
            loss_funct = self._num_mistakes

        errors_pretrain = self.errors_pretrain_ = []
        losses_fine_tune = self.losses_fine_tune_ = []
        errors_fine_tune = self.errors_fine_tune_ = []

        self.momentum = self.pretrain_momentum

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
                            self._minibatches(X_pre),
                            self.epochs_pretrain[layer_index],
                            minibatches_per_epoch,
                        )):
                    errors_pretrain[-1].append(err)
                    if self.verbose:  # pragma: no cover
                        print "  Epoch {}: err {}".format(epoch + 1, err)
                        elapsed = str(timedelta(seconds=time() - time0))
                        print "  ({})".format(elapsed.split('.')[0])
                        print "momentum=" + str(self.momentum)
                        time0 = time()
                    if self.pretrain_callback is not None:
                        self.pretrain_callback(
                            self, epoch + 1, layer_index)

        # # setting another minibatch size
        # self.minibatch_size = 32

        self.momentum = self.fine_tune_momentum

        self._configure_net_finetune(self.net_)
        if self.verbose:  # pragma: no cover
            print "[DBN] Fine-tune..."
        time0 = time()
        acc_log = []
        err = 1.0
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

            acc = "no"
            if len(y_val) > 0:
                # performing validation
                acc = self.score(X_val, y_val)
                # test_acc = self.score(X_test, y_test)
                acc_log.append(acc)
                figure("validation accuracy");
                clf()
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

            if err < 0.1:
                break

            # set high momentum since some epoch
            if epoch < 1:
                self.momentum = 0.0
            else:
                self.momentum = self.fine_tune_momentum

            print "momentum=" + str(self.momentum)

            # if self.verbose:
            #     show()
        return errors_fine_tune, acc_log

    def score(self, X, y):
        loss_funct = self.loss_funct
        if loss_funct is None:
            loss_funct = self._num_mistakes

        outputs = self.predict_proba(X)
        # targets = self._onehot(self._enc.transform(y))
        targets = _make_targets(y, self.layer_sizes[-1])
        mistakes = loss_funct(outputs, targets)
        return - float(mistakes) / len(y) + 1


# class DBN_Model:
def train_and_test(alphabet, feat_labs, context_size=8, file_name='results.txt',
                   n_hidden=3, hidden_size=1024, n_epochs=10, n_epochs_pretrain=0, pre_size=0.8, test_size=0.4,
                   validation_size=0.0, learn_rates=0.1, learn_rate_decays=0.99, momentum=0.9, l2_costs=0.0001,
                   learn_rates_pretrain=0.0001, pretrain_momentum=0.0):
    feat_len = np.shape(feat_labs[0][0])[1]
    input_size = feat_len * (2 * context_size + 1)
    output_size = len(alphabet)
    # output_size = len(np.unique(Y))

    # # layers initializing
    # network = InputLayer((None, input_size))
    # for i in xrange(n_hidden):
    #     network = DenseLayer(network, hidden_size)
    # network = DenseLayer(network, output_size, nonlinearity=softmax)
    #
    # model = nolearn.lasagne.NeuralNet(network, max_epochs=40, update=adam, verbose=1)

    layers = [input_size]
    layers.extend(np.repeat(hidden_size, n_hidden))
    # # botleneck layer
    # layers.append(128)
    layers.append(output_size)
    model = MyDBN(layer_sizes=layers,
                  learn_rates=learn_rates,
                  learn_rate_decays=learn_rate_decays,
                  momentum=momentum,  # set 0.0 during several first epochs
                  l2_costs=l2_costs,
                  epochs=n_epochs,
                  epochs_pretrain=n_epochs_pretrain,
                  pretrain_momentum=pretrain_momentum,
                  learn_rates_pretrain=learn_rates_pretrain,
                  # dropouts=0.3,
                  minibatch_size=128,
                  nesterov=True,
                  verbose=3, )

    length = len(feat_labs)
    pre_length = (1-pre_size)*length
    X_pre, _ = _data_with_context(context_size, input_size, feat_labs)

    feat_labs = feat_labs[:pre_length]

    random_split = cross_validation.ShuffleSplit(len(feat_labs), n_iter=1, test_size=test_size)
    for train_and_valid_index, test_index in random_split:

        train = feat_labs[train_and_valid_index]

        # creating held out validation part in train data
        valid_split = cross_validation.ShuffleSplit(len(train), n_iter=1, test_size=validation_size)
        # X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=validation_size, random_state=0)
        for train_index, valid_index in valid_split:
            X_train, y_train = _data_with_context(context_size, input_size, train[train_index])
            X_val, y_val = _data_with_context(context_size, input_size, train[valid_index])

            errs, accs = model.fit_with_validation(X_pre, X_train, y_train, X_val, y_val, output_size)

        # for testing
        X_test, y_test = _data_with_context(context_size, input_size, feat_labs[test_index])

        if test_size == 0:
            acc = "no"
        else:
            acc = model.score(X_test, y_test)
        print "Test accuracy: " + str(acc)

    # writing results in file
    string = "%d utt., %d cont., DBN%s, %d epochs, LR: %s, mom: %s, LR_decay: %s, L2: %s, %s pretr. epochs, LR_pretr: %s, " \
             "valid. size %r: accuracy=%s,\n  train errors %s\n  valid.accur. %s\n" % (
                 len(feat_labs), context_size, model.layer_sizes, model.epochs, str(model.learn_rates), str(momentum),
                 str(model.learn_rate_decays), str(model.l2_costs), str(model.epochs_pretrain),
                 str(learn_rates_pretrain), validation_size, str(acc), str(errs), str(accs))

    newpath = r'../DBN'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    with open('../DBN/' + file_name, 'a') as file:
        file.write(string)

    return model


def _data_with_context(context_size, input_size, feat_lab):
    letters = set()
    for _, ys in feat_lab:
        for c in ys:
            letters.add(c)

    print "different symbols: " + str(len(letters))

    X = []
    y = []
    for (xs, ys) in feat_lab:
        for i in xrange(context_size, len(xs) - context_size):
            X.append(
                np.ndarray(shape=(input_size,), dtype=float, buffer=xs[i - context_size: i + 1 + context_size]))
            y.append(ys[i])
    # X = np.array(X).astype(np.float32)
    # Y = np.array(Y).astype(np.int32)
    X = np.array(X)
    y = np.array(y)

    from sklearn import preprocessing
    if len(X) > 0:
        X = preprocessing.normalize(X)

    return X, y


def _make_targets(classes, output_size):
    res = []
    for c in classes:
        array = np.zeros((output_size,))
        array[c - 1] = 1
        res.append(array)
    return np.array(res)


def predict(model, (xs, ys), context_size=None):
    X = []
    y = []
    input_size = model.get_params()['layer_sizes'][0]
    output_size = model.get_params()['layer_sizes'][-1]
    feat_len = np.shape(xs)[1]
    if context_size is None:
        context_size = (input_size / feat_len - 1) / 2

    X, y = _data_with_context(context_size, input_size, [(xs, ys)])

    def most_prob(array):
        res = np.zeros((len(array),))
        res[np.argmax(array)] = 1
        return res

    correct = _make_targets(y, output_size)

    pred = model.predict_proba(X)
    # pred = np.array([most_prob(ar) for ar in pred])

    mistakes = model._num_mistakes(pred, correct)
    accuracy = 1.0 - float(mistakes) / len(y)
    print "score: " + str(accuracy)
    # print model.score(X, y)

    # getting frequency of maximal amplitude
    # spec_ys = np.array(spec_ys)
    # assert (ys == spec_ys), "It's bad."
    # freq = np.array([most_prob(freq_vec) for freq_vec in spec_xs])

    figure("prediction")
    clf()
    subplot(211);
    imshow(pred.T, cmap=cm.hot, interpolation='nearest')
    subplot(212);
    imshow(correct.T, cmap=cm.hot, interpolation='nearest')
    # subplot(313); imshow(freq.T, cmap=cm.hot, interpolation='nearest')
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
    model = train_and_test(X_train, y_train, alphabet)
    acc = model.score(X_test, y_test)
    print "accuracy: " + str(acc)
