__author__ = 'larisa'

import numpy as np
import array
from scikits.talkbox.features import mfcc
from sklearn import preprocessing
from features import mfcc as MFCC

class TrainingDataCreator(object):

    def get_feature_vec (self, dataReader):

        frames = []
        labelVec = []
        features = np.zeros((1, 13))
        feat_label_vec = []
        p = 0
        k = 0
        intervals = dataReader.getIntervals()
        for j in range(len(intervals)):
            start = intervals[j]._get_start_time()
            end = intervals[j]._get_end_time()
            step = int((end - start) * dataReader.getFramerate())
            letter = intervals[j].text
            frames_in_interval = []

            for k in range(step):
                frames.append(dataReader.getFrames()[p])
                frames_in_interval.append(dataReader.getFrames()[p])
                p+=1
                labelVec.append(letter)
            ceps = []
            try:
                ceps = mfcc(frames_in_interval)[0]
                #print("frames in int", len(frames_in_interval))
                #print("ceps" , len(ceps))
            except Exception:
                print("ceps=0", dataReader._textGridFile)

            for i in range(len(ceps)):
                normalized_cep = preprocessing.normalize([ceps[i]])
                features = np.append(features, normalized_cep, axis=0)
                feat_label_vec.append(letter)

        features = features[1:,:]
        #print("feat size", len(features))
        #print("lebel size", len(feat_label_vec))
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
        all_features = np.zeros((1, 13))
        all_labels = []

        for i in range(len(self._parsedFiles)):
            (features, feat_label_vec, frames, labelVec) = self.get_feature_vec(self._parsedFiles[i])
            all_features = np.concatenate([all_features, features])
            all_labels = np.concatenate([all_labels, feat_label_vec])

        all_features = all_features[1:,:]
        return (all_features, np.array(all_labels))








