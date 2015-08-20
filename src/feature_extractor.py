__author__ = 'misha'

# import file_rename
# file_rename.rename('/home/misha/Downloads/markups/word_praats')
# file_rename.rename('/home/misha/Downloads/markups/paradigm_praats')

# ---------------------
from textgrid import *
import os

# os.chdir('/home/misha/Downloads/test/markups')
# path = u'.'
# for root, dirs, files in os.walk(path, True):
#     print files
#     name = files[3]
#     file = TextGrid(name)
#     file.read(name)
#     tiers = file.tiers
#     print name
#     for tier in file.tiers:
#         print tier.name
#         for inter in tier.intervals:
#             print '\t'.join([str(inter.mark), str(inter.minTime), str(inter.maxTime)])

import ocrolib

import pylab
import wave
import numpy as np
from scipy.io import wavfile
import os

def extract_specgrams(data):
    intervals = data.getIntervals()
    frames_array = data.getFrames()
    fs = data.getFramerate()
    assert fs == 44100
    nfft = 254
    time_step = 0.5*nfft/fs

    # generate specgram
    Pxx, freqs, t, plot = pylab.specgram(
        frames_array,
        NFFT=nfft,
        Fs=fs,
        detrend=pylab.detrend_none,
        window=pylab.window_hanning,
        # sides='twosided',
        noverlap=int(nfft*0.5))

    # creating sequence of (feature, label)
    feature_label = []
    for interval in intervals:
        start = interval._start_time / time_step
        end = interval._end_time / time_step
        features = Pxx[:, start:end]
        label = interval.text
        for i in xrange(np.shape(features)[1]):
            feature_label.append((features[:, i], label))

    return feature_label



# dir = '/home/misha/Downloads/test/sounds'
# os.chdir(dir)
# path = u'.'
# filename = os.listdir(path)[0]
#
# file = wave.open(filename)
# print "sample width in bytes: " + file.getsampwidth().__str__()
# print "number of audio channels: " + file.getnchannels().__str__()
#
# print "sampling frequency (framerate, frames per second): " + file.getframerate().__str__()
# print "number of audio frames: " + file.getnframes().__str__()
# print file.getparams()
#
# # data = file.readframes(file.getnframes())
# fs, frames = wavfile.read(filename)
#
# channels = [
#     np.array(frames[:, 0]),
#     np.array(frames[:, 1])
# ]
#
# nfft = 254
# print "number of frames in 1 segment: " + nfft.__str__()
#
# # generate specgram
# Pxx, freqs, t, plot = pylab.specgram(
#     channels[0],
#     NFFT=nfft,
#     Fs=fs,
#     detrend=pylab.detrend_none,
#     window=pylab.window_hanning,
#     # sides='twosided',
#     noverlap=int(nfft*0.5))
#
# print "frequencies (" + len(freqs).__str__() + "):"
# print freqs
# print "time points (" + len(t).__str__() + "):"
# print t
# print Pxx[:,0:4]
#
# # pylab.show()
# file.close()

# ------------------------------

# import neuralnetwork
# import reberGrammar
#
# train_data = reberGrammar.get_n_embedded_examples(1000)
#
# learn_rnn_fn = theano.function(inputs=[neuralnetwork.v, neuralnetwork.target],
#                                outputs=neuralnetwork.cost,
#                                updates=neuralnetwork.updates)
#
# nb_epochs = 250
# train_errors = np.ndarray(nb_epochs)
#
#
# def train_rnn(train_data):
#     for x in range(nb_epochs):
#         error = 0.
#         for j in range(len(train_data)):
#             index = np.random.randint(0, len(train_data))
#             i, o = train_data[index]
#             train_cost = learn_rnn_fn(i, o)
#             error += train_cost
#         train_errors[x] = error
#
#
# train_rnn(train_data)
#
# # %matplotlib inline
# import matplotlib.pyplot as plt
#
# plt.plot(np.arange(nb_epochs), train_errors, 'b-')
# plt.xlabel('epochs')
# plt.ylabel('error')
# plt.ylim(0., 50)

