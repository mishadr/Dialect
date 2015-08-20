__author__ = 'larisa'

#ecoding: utf8

from dataparser import *
from training_data_creator import *
from nn import *

import os
import fnmatch
import feature_extractor

def get_files(path, extension):
    tgs = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, extension):
            tgs.append(os.path.join(root, filename))

    tgs.sort()
    return tgs

if __name__ == "__main__":
    #reload(sys)

    markups_path = '/home/misha/Downloads/test/markups/'
    sounds_path = '/home/misha/Downloads/test/sounds/'

    # reading data from files
    textGrids = get_files(markups_path, '*.TextGrid')
    sounds = get_files(sounds_path, '*.wav')
    if len(textGrids) != len(sounds):
        raise Exception("Number of sound files and number of TextGrid files are different ")

    print str(len(textGrids)) + " files read"

    # parsing data
    parsedFiles = []
    for i in range(len(sounds)):
        parsedData = DataReader(sounds[i], textGrids[i])
        if parsedData.is_correct():
            parsedFiles.append(parsedData)

    print str(len(parsedFiles)) + " files correct"

    feat_lab = feature_extractor.extract_specgrams(parsedFiles[0])
    print feat_lab

    # training data
    training_data_creator = TrainingDataCreator(parsedFiles)
    (train, labels) = training_data_creator.get_training_data()
    print("trains", len(train))
    print("labels", len(labels))
    alphabet = training_data_creator.get_alphabet()
    print "alphabet size: " + str(len(alphabet))

    accuracy = oneholdout(train, labels, alphabet)
    print(accuracy)

