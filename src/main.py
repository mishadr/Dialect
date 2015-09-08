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

    # FIXME
    # I can't understand what's wrong with this particular files (in DBN learning only), but
    try:
        parsedFiles.pop(17)
        parsedFiles.pop(211)
        parsedFiles.pop(211)
        parsedFiles.pop(211)
        parsedFiles.pop(211)
    except IndexError:
        pass

    print str(len(parsedFiles)) + " utterances will be used"
    feature_extractor = FeatureExtractor(parsedFiles)
    alphabet = feature_extractor.get_alphabet()
    print str(len(alphabet)) + u','.join(alphabet.keys())
    feat_labs = array(feature_extractor.get_mfcc_features())

    # print "target sequences:"
    # for p in parsedFiles:
    #     print ([alphabet[s.text] for s in p.getIntervals()])

    # # saving features in file
    # with open("feat.db", 'w') as file:
    #     pickle.dump((alphabet, feat_labs), file)

    # ----------------- DBN training
    # dbn_trainer.train(alphabet, feat_labs, context_size=7, n_hidden=4, hidden_size=1024, file_name="bigtests")
    # for i in xrange(4, 13):
    #     try:
    #         dbn_trainer.train(alphabet, feat_labs, context_size=i, n_hidden=4, hidden_size=1024)
    #     except Exception:
    #         print "At i=%d failed" % i
    for n in xrange(1, 5):
        for size in [256, 512, 1024, 2048]:
            try:
                dbn_trainer.train(alphabet, feat_labs, context_size=9, n_hidden=n, hidden_size=size)
            except Exception:
                print "At %s, %s failed" % (n, size)


    # training_data_creator = TrainingDataCreator(parsedFiles)
    # (X, Y) = training_data_creator.get_training_data()
    # print("trains", len(X))
    # print("labels", len(Y))
    # alphabet = training_data_creator.get_alphabet()
    # print "alphabet size: " + str(len(alphabet))
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    # model = train(X_train, y_train, alphabet)
    # acc = model.score(X_test, y_test)
    # print "accuracy: " + str(acc)
    #
    # test_x = X[-10:]
    # test_y = Y[-10:]
    #
    # print model.predict(test_x)
    # print test_y


    # # ----------------- LSTM training
    # print [np.shape(xs) for (xs, cs) in feat_labs]
    #
    # lstm_trainer.train(alphabet, feat_labs)
