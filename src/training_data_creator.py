__author__ = 'larisa'

import numpy as np
from scikits.talkbox.features import mfcc
from sklearn import preprocessing


class TrainingDataCreator(object):
    """ Training data: sequence of feature vectors and sequence of corresponding IFA symbols
    """

    def get_feature_vec(self, data):

        frames = []
        labelVec = []
        features = np.zeros((1, 13))
        feat_label_vec = []
        p = 0
        for interval in data.getIntervals():
            start = interval._get_start_time()
            end = interval._get_end_time()
            step = int((end - start) * data.getFramerate())
            letter = interval.text
            frames_in_interval = []

            for k in range(step):
                frames.append(data.getFrames()[p])
                frames_in_interval.append(data.getFrames()[p])
                p += 1
                labelVec.append(letter)
            ceps = []
            try:
                ceps = mfcc(frames_in_interval)[0]
                # print("frames in int", len(frames_in_interval))
                # print("ceps" , len(ceps))
            except Exception:
                print("ceps=0", data._textGridFile)

            for i in range(len(ceps)):
                normalized_cep = preprocessing.normalize([ceps[i]])
                features = np.append(features, normalized_cep, axis=0)
                feat_label_vec.append(letter)

        features = features[1:, :]
        # print("feat size", len(features))
        # print("lebel size", len(feat_label_vec))
        return (features, np.array(feat_label_vec), frames, labelVec)

    def __init__(self, parsedFiles):
        self._parsedFiles = parsedFiles

    def get_alphabet(self):
        letters = []
        for i in range(len(self._parsedFiles)):
            intervals = self._parsedFiles[i].getIntervals()
            for j in range(len(intervals)):
                letters.append(intervals[j].text)

        alphabet = list(np.unique(letters))

        return alphabet

    def get_training_data(self):
        """ Gets training data
        :return:
        """
        all_features = np.zeros((1, 13))
        all_labels = []

        for data in self._parsedFiles:
            (features, feat_label_vec, frames, labelVec) = self.get_feature_vec(data)
            all_features = np.concatenate([all_features, features])
            all_labels = np.concatenate([all_labels, feat_label_vec])

        all_features = all_features[1:, :]
        return all_features, np.array(all_labels)
