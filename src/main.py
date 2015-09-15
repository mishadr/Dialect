from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

__author__ = 'larisa'

#ecoding: utf8

from dataparser import *
from training_data_creator import *
import dbn_trainer
import lstm_trainer
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
    n_utterances = 3000
    parsedFiles = []
    n_correct = 0
    for i in range(len(textGrids)):
        parsedData = DataReader(sounds[i], textGrids[i])
        if parsedData.is_correct():
            parsedFiles.append(parsedData)
            n_correct += 1
        if n_correct >= n_utterances:
            break

    # FIXME
    # I can't understand what's wrong with this particular files (DBN learning fails)
    try:
        parsedFiles.pop(17)
        parsedFiles.pop(211)
        parsedFiles.pop(211)
        parsedFiles.pop(211)
        parsedFiles.pop(211)
    except IndexError:
        pass

    # to unify alphabet encoding
    parsedFiles.sort()

    print str(len(parsedFiles)) + " utterances will be used"
    feature_extractor = FeatureExtractor(parsedFiles)
    alphabet = feature_extractor.get_alphabet()
    print str(len(alphabet)) + " symbols in alphabet: " + u','.join(alphabet.keys())
    feat_labs = array(feature_extractor.get_mfcc_features())
    # print [np.shape(xs) for (xs, cs) in feat_labs]
    # spec_feat = feature_extractor.get_spec_features()
    

    # print "target sequences:"
    # for p in parsedFiles:
    #     print ([alphabet[s.text] for s in p.getIntervals()])

    # # saving features in file
    # with open("feat.db", 'w') as file:
    #     pickle.dump((alphabet, feat_labs), file)

    # ----------------- DBN training
    # for i in xrange(1, 15):
    #     try:
    #         dbn_trainer.train(alphabet, feat_labs, context_size=7, n_hidden=4, n_epochs=12, n_epochs_pretrain=i)
    #     except Exception:
    #         print "At i=%d failed" % i

    # for i in xrange(4, 13):
    #     try:
    #         dbn_trainer.train(alphabet, feat_labs, context_size=i, n_hidden=4, hidden_size=1024)
    #     except Exception:
    #         print "At i=%d failed" % i

    # for n in xrange(5, 6):
    #     for size in [256, 512, 1024, 2048]:
    #         try:
    #             dbn_trainer.train(alphabet, feat_labs, context_size=9, n_hidden=n, hidden_size=size)
    #         except Exception:
    #             print "At %s, %s failed" % (n, size)

    # random_split = cross_validation.ShuffleSplit(len(feat_labs), n_iter=1, test_size=0.2, random_state=13)
    # for train_index, test_index in random_split:

    # # training
    # model = dbn_trainer.train(alphabet, feat_labs, n_epochs=15, context_size=7, n_hidden=4, validation_on=False)
    #
    # # # predicting
    # # for f in feat_labs[90:99]:
    # #     dbn_trainer.predict(model, f)
    #
    # # saving model to file
    # newpath = r'../models'
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
    # with open("../models/dbn_model_3000.7f", 'w') as file:
    #     pickle.dump(model, file)

    # reading model from file
    with open("../models/dbn_model_3000.7", 'r') as file:
        model = pickle.load(file)

    acc_log = []
    for f in feat_labs[2980:]:
        acc = dbn_trainer.predict(model, f)
        acc_log.append(acc)

    print 1.0*sum(np.array(acc_log))/len(acc_log)

    # # ----------------- LSTM training
    #
    # lstm_trainer.train(alphabet, feat_labs)
