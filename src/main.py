from __builtin__ import sum
from gi.overrides.GLib import get_current_time
from scipy.constants.constants import alpha

__author__ = 'larisa'

#ecoding: utf8

from dataparser import *
from training_data_creator import *
from nn import *
import pickle

import pdb
import os
import fnmatch
from feature_extractor import *
from ocrolib.lstm import *
# import kaldi.decoders

class MyParallel(Parallel):
    """Difference is that backward function returns deltas.
    """
    # def __init__(self,*nets):
    #     self.nets = netsb.s
    def backward(self,deltas):
        deltas = array(deltas)
        new_deltas = []
        start = 0
        for i,net in enumerate(self.nets):
            k = net.noutputs()
            new_deltas.append(np.array(net.backward(deltas[:,start:start+k])))
            start += k

        # I'm not sure we should sum the deltas from parallel layers
        res = new_deltas[0]
        for i in range(1, len(self.nets)):
            res += new_deltas[i]
        return res


class MySeqRecognizer(SeqRecognizer):
    def trainSequence(self,xs,cs,update=1,key=None):
        "Train with an integer sequence of codes."
        assert xs.shape[1]==self.Ni,"wrong image height"
        # forward step
        self.outputs = array(self.lstm.forward(xs))
        # CTC alignment
        self.targets = array(make_target(cs,self.No))
        self.aligned = array(ctc_align_targets(self.outputs,self.targets,debug=1))
        # propagate the deltas back
        deltas = self.aligned-self.outputs
        self.lstm.backward(deltas)
        if update: self.lstm.update()
        # translate back into a sequence
        result = translate_back(self.outputs, threshold=0.1)
        # compute least square error
        self.error = sum(deltas**2)
        self.error_log.append(self.error**.5/len(cs))
        # compute class error
        self.cerror = edist.levenshtein(cs,result)
        self.cerror_log.append((self.cerror,len(cs)))
        # training keys
        self.key_log.append(key)
        return result


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
    parsedFiles = []
    for i in range(750):
        parsedData = DataReader(sounds[i], textGrids[i])
        if parsedData.is_correct():
            parsedFiles.append(parsedData)

    print str(len(parsedFiles)) + " files correctly marked up"

    # # saving features in file
    # with open("feat.db", 'w') as file:
    #     pickle.dump((alphabet, feat_labs), file)

    # # ----------------- DBN training
    # training_data_creator = TrainingDataCreator(parsedFiles)
    # (X, Y) = training_data_creator.get_training_data()
    # print("trains", len(X))
    # print("labels", len(Y))
    # alphabet = training_data_creator.get_alphabet()
    # print "alphabet size: " + str(len(alphabet))
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    # model = train(X_train, y_train, alphabet)
    # acc = model.score(X_test, y_test)
    # print "accuracy: " + str(acc)
    #
    # test_x = X[-10:]
    # test_y = Y[-10:]
    #
    # print model.predict(test_x)
    # print test_y


    # extracting specgram features
    feature_extractor = FeatureExtractor(parsedFiles)
    alphabet = feature_extractor.get_alphabet()
    print str(len(alphabet)) + u','.join(alphabet.keys())
    feat_labs = array(feature_extractor.get_mfcc_features())
    print [np.shape(xs) for (xs, cs) in feat_labs]

    # ---------------- LSTM learning

    Ni = 13
    Ns = 500
    No = len(alphabet)
    model = Stacked([LSTM(Ni, Ns), Softmax(Ns, No)])
    # model = Stacked([MyParallel(LSTM(Ni, Ns), Reversed(LSTM(Ni, Ns))),
    #                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
    #                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
    #                            Softmax(2*Ns, No)])
    model.setLearningRate(0.0001, 0.9)

    print "target sequences:"
    for p in parsedFiles:
        print ([alphabet[i.text] for i in p.getIntervals()])

    def make_targets(classes):
        res = []
        for c in classes:
            array = zeros((No,))
            array[c-1] = 1
            res.append(array)
        return res


    # splitting randomly into train and test datasets
    random_split = cross_validation.ShuffleSplit(len(feat_labs), n_iter=10000, test_size=0.1, random_state=0)
    train_sq_error_log = []
    test_sq_error_log = []
    iteration = 0
    for train_index, test_index in random_split:
        print "iteration " + str(iteration)

        # training
        lev_dist = 0
        sq_error = []
        for (xs, cs) in feat_labs[train_index]:
            ys = np.array(make_targets(cs))
            answer = model.train(np.array(xs), ys)
            sq_error.append(sum((ys-answer)**2)/len(cs))
            pred = argmax(answer, axis=1)+1
            # print ", ".join([str(pred), str(cs)])
            lev_dist += edist.levenshtein(cs, pred)
        print "TRAIN. total levenshtein dist: " + str(lev_dist)
        print "TRAIN. square error: " + str(sq_error)
        train_sq_error_log.append(sum(sq_error))

        figure("sum of errors. 388+43 utter, LR(0.0001, 0.9), LSTM(13, 500, 84)"); clf();
        subplot(211)
        plot(array(train_sq_error_log), "r")
        # plot(ndarray((len(sq_error), ), dtype=float, buffer=array(sq_error)), "r")

        # testing
        if iteration%1 == 0:
            lev_dist = 0
            sq_error = []
            for (xs, cs) in feat_labs[test_index]:
                ys = np.array(make_targets(cs))
                answer = model.predict(xs)
                sq_error.append(sum((ys-answer)**2)/len(cs))
                pred = argmax(answer, axis=1)+1
                lev_dist += edist.levenshtein(cs, pred)
            print "TEST. total levenshtein dist: " + str(lev_dist)
            print "TEST. square error: " + str(sq_error)
            test_sq_error_log.append(sum(sq_error))

        subplot(212)
        plot(array(test_sq_error_log), "g")
        ginput(1,0.01);
        iteration += 1

        # if i%1000 == 999:
        #     # saving model in file
        #     model.dstats = None
        #     model.ldeltas = None
        #     model.deltas = None
        #     with open("../models/lstm_model_" + str(get_current_time()), 'w') as file:
        #         pickle.dump(model, file)

    # # ---------------- LSTM learning with CTC alignment
    #
    # Ni = 128
    # Ns = 100
    # No = len(alphabet)+1
    # model = MySeqRecognizer(Ni, Ns, No)
    #
    # # model.lstm = Stacked([MyParallel(LSTM(Ni, Ns), Reversed(LSTM(Ni, Ns))),
    # #                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
    # #                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
    # #                            Softmax(2*Ns, No)])
    # model.lstm = Stacked([LSTM(Ni, Ns), LSTM(Ns, Ns),
    #                       Softmax(Ns, No)])
    # model.setLearningRate(0.001)
    #
    # print "target sequences:"
    # for p in parsedFiles:
    #     print ([alphabet[i.text] for i in p.getIntervals()])
    #
    # for i in xrange(200):
    #     print "iteration " + str(i)
    #     for (xs, cs) in feat_labs:
    #         answer = model.trainSequence(np.array(xs), np.array(cs))
    #         print ", ".join([str(answer), str(cs), "|error gradient|="+str(model.error)])
    #         # print model.outputs


