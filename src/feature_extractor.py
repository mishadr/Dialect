__author__ = 'misha'

import pylab
import numpy as np
from scikits.talkbox.features import mfcc
from sklearn import preprocessing


class FeatureExtractor:
    """ Container for feature to label mapping and label alphabet
    extracted from given marked up files.
    """

    def __init__(self, parsedFiles):
        self.alphabet = self.extract_alphabet(parsedFiles)
        self.feature_label = []
        self.parsed_files = parsedFiles
        self.mfcc_feature_label = None
        self.spec_feature_label = None

    def extract_alphabet(self, parsedFiles):
        letters = []
        for file in parsedFiles:
            for interval in file.getIntervals():
                letters.append(interval.text)

        alph = list(np.unique(letters))
        dict = {}
        for i in xrange(len(alph)):
            dict[alph[i]] = i+1

        return dict

    def extract_spectrogram_features(self, data):
        intervals = data.getIntervals()
        frames_array = data.getFrames()
        fs = data.getFramerate()
        if fs != 44100:
            print "different framerate: "+str(fs)
        nfft = 254
        time_step = 0.5*nfft/fs

        # generating specgram
        Pxx, freqs, t, plot = pylab.specgram(
            frames_array,
            NFFT=nfft,
            Fs=fs,
            detrend=pylab.detrend_none,
            window=pylab.window_hanning,
            # sides='twosided',
            noverlap=int(nfft*0.5))

        # creating sequence of (feature, label)
        xmax = np.max(Pxx)
        feature_label = ([], [])
        for n, interval in enumerate(intervals):
            start = interval._start_time / time_step
            end = interval._end_time / time_step
            features = Pxx[:, start:end]
            label = interval.text
            for i in xrange(np.shape(features)[1]):
                # FIXME
                # I'm not sure whether we should normalize it like that!!!
                feature_label[0].append(features[:, i]/xmax)
                feature_label[1].append(self.alphabet[label])

        # if np.shape(feature_label[0])[1] != 128:
        #     print "short interval"
        return feature_label

    def extract_mfcc_feature_vec(self, data):
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
                frames_in_interval.append(data.getFrames()[p])
                p += 1
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
                feat_label_vec.append(self.alphabet[letter])

        features = features[1:, :]
        # print("feat size", len(features))
        # print("lebel size", len(feat_label_vec))
        return (features, np.array(feat_label_vec))

    def get_alphabet(self):
        return self.alphabet

    def get_mfcc_features(self):
        if self.mfcc_feature_label is None:
            self.mfcc_feature_label = []
            for data in self.parsed_files:
                self.mfcc_feature_label.append(self.extract_mfcc_feature_vec(data))

        return self.mfcc_feature_label

    def get_spec_features(self):
        if self.spec_feature_label is None:
            self.spec_feature_label = []
            for data in self.parsed_files:
                self.spec_feature_label.append(self.extract_spectrogram_features(data))

        return self.spec_feature_label



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

