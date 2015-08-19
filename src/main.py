__author__ = 'larisa'

#ecoding: utf8

from dataparser import *
from training_data_creator import *
from nn import *

import sys
import os
import fnmatch

def get_files(path, extention):
    tgs = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, extention):
            tgs.append(os.path.join(root, filename))

    return tgs

if __name__ == "__main__":
    #reload(sys)

    markups_path = '/Users/larisa/Downloads/markups/word_praats/sounds/'
    sounds_path = '/Users/larisa/Downloads/markups/word_praats/markups/'

    #markups_path = '/Users/larisa/markups/'
    #sounds_path = '/Users/larisa/markups/'

    textGrids = get_files(markups_path, '*.wav')
    sounds = get_files(sounds_path, '*.TextGrid')
    print(len(textGrids))
    if len(textGrids) != len(sounds):
        raise Exception("Number of sound files and number of TextGrid files are different ")

    parsedFiles = []
    for i in range(len(sounds)):
        parsedData = DataReader(sounds[i], textGrids[i])
        if parsedData.getErr() == 0:
            parsedFiles.append(parsedData)

    training_data_creator = TrainingDataCreator(parsedFiles)
    (train, labels) = training_data_creator.get_training_data()
    print("trains", len(train))
    print("labels", len(labels))
    alphabet = training_data_creator.get_alphabet()
    print(len(alphabet))

    accuracy = oneholdout(train, labels, alphabet)
    print(accuracy)

