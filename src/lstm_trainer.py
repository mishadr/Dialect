__author__ = 'misha'

from sklearn import cross_validation
from ocrolib.lstm import *


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
        self.aligned = array(ctc_align_targets(self.outputs,self.targets))
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


def train(alphabet, feat_labs):

    # splitting randomly into train and test datasets
    random_split = cross_validation.ShuffleSplit(len(feat_labs), n_iter=10000, test_size=0.10, random_state=0)


    # # ---------------- LSTM learning
    #
    # Ni = 13
    # Ns = 500
    # No = len(alphabet)
    # model = Stacked([LSTM(Ni, Ns), Softmax(Ns, No)])
    # # model = Stacked([MyParallel(LSTM(Ni, Ns), Reversed(LSTM(Ni, Ns))),
    # #                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
    # #                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
    # #                            Softmax(2*Ns, No)])
    # model.setLearningRate(0.00001, 0.5)
    #
    # def make_targets(classes):
    #     res = []
    #     for c in classes:
    #         array = zeros((No,))
    #         array[c-1] = 1
    #         res.append(array)
    #     return res
    #
    #
    # train_sq_error_log = []
    # test_sq_error_log = []
    # iteration = 0
    # for train_index, test_index in random_split:
    #     print "iteration " + str(iteration)
    #
    #     # training
    #     lev_dist = 0
    #     total_dist = 0
    #     sq_error = []
    #     for (xs, cs) in feat_labs[train_index]:
    #         ys = np.array(make_targets(cs))
    #         answer = model.train(np.array(xs), ys)
    #         sq_error.append(sum((ys-answer)**2)/len(cs))
    #         pred = argmax(answer, axis=1)+1
    #         # print ", ".join([str(pred), str(cs)])
    #         lev_dist += edist.levenshtein(cs, pred)
    #         total_dist += len(cs)
    #     accuracy = 100 - 100.0 * lev_dist/total_dist
    #     print "TRAIN. total levenshtein dist: " + str(lev_dist) + "\t accuracy: " + str(accuracy)
    #     print "TRAIN. square error: " + str(sq_error)
    #     train_sq_error_log.append(sum(sq_error))
    #
    #     figure("sum of errors. 10, 13 500 21"); clf();
    #     subplot(211)
    #     plot(array(train_sq_error_log), "r")
    #     # plot(ndarray((len(sq_error), ), dtype=float, buffer=array(sq_error)), "r")
    #
    #     # testing
    #     if iteration%2 == 0:
    #         lev_dist = 0
    #         total_dist = 0
    #         sq_error = []
    #         for (xs, cs) in feat_labs[test_index]:
    #             ys = np.array(make_targets(cs))
    #             answer = model.predict(xs)
    #             sq_error.append(sum((ys-answer)**2)/len(cs))
    #             pred = argmax(answer, axis=1)+1
    #             lev_dist += edist.levenshtein(cs, pred)
    #             total_dist += len(cs)
    #         accuracy = 100 - 100.0 * lev_dist/total_dist
    #         print "TEST. total levenshtein dist: " + str(lev_dist) + "\t accuracy: " + str(accuracy)
    #         print "TEST. square error: " + str(sq_error)
    #         test_sq_error_log.append(sum(sq_error))
    #
    #     subplot(212)
    #     plot(array(test_sq_error_log), "g")
    #     ginput(1,0.01);
    #     iteration += 1
    #
    #     # if i%1000 == 999:
    #     #     # saving model in file
    #     #     model.dstats = None
    #     #     model.ldeltas = None
    #     #     model.deltas = None
    #     #     with open("../models/lstm_model_" + str(get_current_time()), 'w') as file:
    #     #         pickle.dump(model, file)

    # ---------------- LSTM learning with CTC alignment

    Ni = 13
    Ns = 300
    No = len(alphabet)+1
    model = MySeqRecognizer(Ni, Ns, No)

    # model.lstm = Stacked([MyParallel(LSTM(Ni, Ns), Reversed(LSTM(Ni, Ns))),
    #                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
    #                            MyParallel(LSTM(2*Ns, Ns), Reversed(LSTM(2*Ns, Ns))),
    #                            Softmax(2*Ns, No)])
    model.lstm = Stacked([MyParallel(LSTM(Ni, Ns), Reversed(LSTM(Ni, Ns))),
                          Softmax(2*Ns, No)])
    model.setLearningRate(0.0001, 0.9)

    train_sq_error_log = []
    test_sq_error_log = []
    train_lev_dist_log = []
    test_lev_dist_log = []
    iteration = 0
    for train_index, test_index in random_split:
        print "iteration " + str(iteration)

        # training
        lev_dist = 0
        total_dist = 0
        sq_error = []
        for (xs, cs) in feat_labs[train_index]:
            pred = model.trainSequence(np.array(xs), np.array(cs))
            sq_error.append(model.error/len(cs))
            lev_dist += edist.levenshtein(cs, pred)
            # total_dist += len(cs)
        # accuracy = 100 - 100.0 * lev_dist/total_dist
        print "TRAIN. total levenshtein dist: " + str(lev_dist)
        print "TRAIN. square error: " + str(sq_error)
        train_sq_error_log.append(sum(sq_error))
        train_lev_dist_log.append(lev_dist)

        figure("sum of squared deltas. 7+3, 13 300-300 21 CTC"); clf();
        subplot(311)
        plot(array(train_sq_error_log), "r")
        subplot(312)
        plot(array(train_lev_dist_log), "b")
        # plot(ndarray((len(sq_error), ), dtype=float, buffer=array(sq_error)), "r")

        # testing
        if iteration%2 == 0:
            lev_dist = 0
            total_dist = 0
            sq_error = []
            for (xs, cs) in feat_labs[test_index]:
                pred = model.trainSequence(np.array(xs), np.array(cs))
                sq_error.append(model.error/len(cs))
                lev_dist += edist.levenshtein(cs, pred)
            print "TEST. total levenshtein dist: " + str(lev_dist)
            print "TEST. square error: " + str(sq_error)
            test_sq_error_log.append(sum(sq_error))
            test_lev_dist_log.append(lev_dist)

        subplot(313)
        plot(array(test_lev_dist_log), "g")
        ginput(1,0.01);
        iteration += 1

    # for i in xrange(200):
    #     print "iteration " + str(i)
    #     for (xs, cs) in feat_labs:
    #         answer = model.trainSequence(np.array(xs), np.array(cs))
    #         print ", ".join([str(answer), str(cs), "|error gradient|="+str(model.error)])
    #         # print model.outputs

