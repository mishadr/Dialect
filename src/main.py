from __builtin__ import xrange
from lasagne.updates import momentum
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import hmm

__author__ = 'larisa'

#ecoding: utf8

from dataparser import *
from training_data_creator import *
import dbn_trainer
import lstm_trainer
import tf_feedforward
import tf_seq2seq
import pickle

import pdb
import fnmatch
from feature_extractor import *
from ocrolib.lstm import *
# import kaldi.decoders

def get_files(path, extension):
    tgs = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, extension):
            tgs.append(os.path.join(root, filename))

    tgs.sort()
    return tgs

if __name__ == "__main__":
    #reload(sys)

    markups_path = '/home/misha/Downloads/markups/word_praats/markups/'
    sounds_path = '/home/misha/Downloads/markups/word_praats/sounds/'
    newpath = r'../models'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # reading data from files
    textGrids = get_files(markups_path, '*.TextGrid')
    sounds = get_files(sounds_path, '*.wav')
    if len(textGrids) != len(sounds):
        raise Exception("Number of sound files and number of TextGrid files are different ")

    print str(len(textGrids)) + " files read"

    # parsing data
    n_utterances = 100
    parsedFiles = []
    n_correct = 0
    for i in range(len(textGrids)):
        parsedData = DataReader(sounds[i], textGrids[i])
        if parsedData.is_correct():
            parsedFiles.append(parsedData)
            n_correct += 1
        if n_correct >= n_utterances:
            break

    # to unify alphabet encoding
    parsedFiles.sort()

    # # saving general alphabet (taken from all files)
    # alphabet = FeatureExtractor(parsedFiles).get_alphabet()
    # with open("../models/alphabet.194", 'w') as file:
    #     pickle.dump(alphabet, file)

    # loading general alphabet
    with open("../models/alphabet.194", 'r') as file:
        alphabet = pickle.load(file)

    print str(len(alphabet)) + " symbols in alphabet:"
    print u', '.join([u"{0:s} = {1:d}".format(sym, id) for sym, id in alphabet.items()])

    print str(len(parsedFiles)) + " utterances will be used"
    feature_extractor = FeatureExtractor(parsedFiles)

    # spec_feat = feature_extractor.get_spec_features()
    # print [np.shape(xs) for (xs, cs) in spec_feat]

    feat_labs = array(feature_extractor.get_spec_features(alphabet))
    print [np.shape(xs) for (xs, cs) in feat_labs]


    # print "target sequences:"
    # for p in parsedFiles:
    #     print ([alphabet[s.text] for s in p.getIntervals()])

    # # saving features in file
    # with open("feat.db", 'w') as file:
    #     pickle.dump((alphabet, feat_labs), file)


    # # ---------------------- HMM training
    #
    # startprob = np.array([0.6, 0.3, 0.1])
    # transmat = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
    # means = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    # covars = np.tile(np.identity(2), (3, 1, 1))
    # model = hmm.GaussianHMM(3, "full", startprob, transmat)
    # model.means_ = means
    # model.covars_ = covars
    # X, Z = model.sample(100)


    # # ----------------- TensorFlow training

    # tf_feedforward.train_and_test(alphabet, feat_labs)

    tf_seq2seq.train_and_test(alphabet, feat_labs)

    # ----------------- DBN training
    # for i in xrange(7, 10):
    #     try:
    #         dbn_trainer.train_and_test(alphabet, feat_labs, context_size=i, n_epochs=10, n_hidden=1,
    #                                    learn_rates=0.001, validation_size=0.3, test_size=0.0)
    #     except Exception:
    #         print "At i=%d failed" % i

    # for n in [3, 4, 5]:
    #     for size in [512, 1024, 2048]:
    #         try:
    #             dbn_trainer.train_and_test(alphabet, feat_labs, context_size=7, n_epochs=25, n_hidden=n, hidden_size=size,
    #                                        learn_rates=0.01, momentum=0.9, validation_size=0.3, test_size=0.0)
    #         except Exception:
    #             print "At i=%d failed" % size

    # for lrd in [1.0, 0.99, 0.9, 0.8]:
    #     try:
    #         dbn_trainer.train_and_test(alphabet, feat_labs, context_size=8, n_epochs=15,
    #                                    learn_rates=0.01, learn_rate_decays=lrd, validation_size=0.3, test_size=0.0)
    #     except Exception:
    #         print "At i=%d failed" % lrd

    # for l in [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]:
    #     try:
    #         dbn_trainer.train_and_test(alphabet, feat_labs, context_size=7, n_epochs=25,
    #                                    learn_rates=0.01, momentum=0.9, l2_costs=l, validation_size=0.3, test_size=0.0)
    #     except Exception:
    #         print "At i=%d failed" % l

    # for m in [0.5, 0.9]:
    #     try:
    #         dbn_trainer.train_and_test(alphabet, feat_labs, context_size=7, n_epochs=300, n_hidden=4, hidden_size=size,
    #                                    n_epochs_pretrain=15, learn_rates_pretrain=0.0001, pretrain_momentum=m,
    #                                    learn_rates=0.03, momentum=0.5, l2_costs=0.000, learn_rate_decays=1.0,
    #                                    validation_size=0.3, test_size=0.0)
    #     except Exception:
    #         print "At i=%d failed" % m

    # model = dbn_trainer.train_and_test(alphabet, feat_labs, context_size=7, n_epochs=20,
    #                                    n_epochs_pretrain=6, learn_rates_pretrain=0.0001, pretrain_momentum=0.0,
    #                                    learn_rates=0.01, momentum=0.9, l2_costs=0.0001,
    #                                    validation_size=0.3, test_size=0.0)

    # feats = [a for a, b in feat_labs]
    #
    # # # reading model from file
    # # with open("../models/dbn_model_pre_2500.194", 'r') as file:
    # #     model = pickle.load(file)
    #
    # context_size = 7
    #
    # feat_len = np.shape(feat_labs[0][0])[1]
    # input_size = feat_len * (2 * context_size + 1)
    # output_size = len(alphabet)

    # model = dbn_trainer.DBN_Model(input_size, output_size, context_size=context_size)
    #
    # model = dbn_trainer.pretrain(model, feats, n_epochs_pretrain=2, learn_rates_pretrain=0.0001, pretrain_momentum=0.0)

    # # saving model to file
    # newpath = r'../models'
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
    # with open("../models/dbn_model_pre_2500.194", 'w') as file:
    #     pickle.dump(model, file)

    # model = dbn_trainer.train_and_test(model, feat_labs, validation_size=0.3, test_size=0.0,
    #                                    n_epochs=20, learn_rates=0.01, momentum=0.9, l2_costs=0.0001)

    # random_split = cross_validation.ShuffleSplit(len(feat_labs), n_iter=1, test_size=0.4, random_state=13)
    # for train_index, test_index in random_split:
    #
    #     # training
    #     model = dbn_trainer.train_and_test(alphabet, feat_labs[train_index], n_epochs=15, test_size=0.4, validation_size=0.2)
    #
    #     # saving model to file
    #     newpath = r'../models'
    #     if not os.path.exists(newpath):
    #         os.makedirs(newpath)
    #     with open("../models/dbn_model_100.1", 'w') as file:
    #         pickle.dump(model, file)
    #
    #     # # reading model from file
    #     # with open("../models/dbn_model_100.1", 'r') as file:
    #     #     model = pickle.load(file)
    #
    #     # predicting
    #     for i in test_index:
    #         f, spec = feat_labs[i], spec_feat[i]
    #         dbn_trainer.predict(model, f, spec)

    # model = dbn_trainer.train_and_test(alphabet, feat_labs, n_epochs=15, test_size=0.1, validation_size=0.4)

    # # reading model from file
    # with open("../models/dbn_model_100.1", 'r') as file:
    #     model = pickle.load(file)
    #
    # acc_log = []
    # for i in xrange(0, 40):
    #     f = feat_labs[i]
    #     # spec = spec_feat[i]
    #     acc = dbn_trainer.predict(model, f)
    #     acc_log.append(acc)
    #
    # print mean(acc_log)

    # # ----------------- LSTM training
    #
    # lstm_trainer.train(alphabet, feat_labs)


