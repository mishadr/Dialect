from matplotlib import cm
from scipy.signal import butter, lfilter, freqz

__author__ = 'misha'

import pylab
import numpy as np
from scikits.talkbox.features import mfcc
from sklearn import preprocessing

import features as pspf


class FeatureExtractor:
    """ Container for feature to label mapping and label alphabet
    extracted from given marked up files.
    """

    def __init__(self, parsedFiles):
        self.alphabet = self.extract_alphabet(parsedFiles)
        self.feature_label = []
        self.parsed_files = parsedFiles
        self._mfcc_feature_label = None
        self.spec_feature_label = None

    def extract_alphabet(self, parsedFiles):
        letters = []
        letters_counter = {}
        for file in parsedFiles:
            for interval in file.getIntervals():
                let = interval.text
                letters.append(let)
                if let in letters_counter:
                    letters_counter[let] += 1
                else:
                    letters_counter[let] = 1

        letters_stat = {}
        for key, count in letters_counter.items():
            if count in letters_stat:
                letters_stat[count].append(key)
            else:
                letters_stat[count] = [key]

        # self.exclude = letters_stat[1]

        # for count, items in letters_stat.items():
        #     print u"%d times of: %s" % (count, u', '.join(items))

        alph = list(np.unique(letters))
        dict = {}
        for i in xrange(len(alph)):
            dict[alph[i]] = i + 1

        return dict

    def extract_spectrogram_features(self, data, alphabet):
        intervals = data.getIntervals()
        frames_array = data.getFrames()
        fs = data.getFramerate()
        if fs != 44100:
            print "different framerate: " + str(fs)
        nfft = 1022
        time_step = 0.5 * nfft / fs

        # cut voice frequencies of [300, 4000] Hz
        # low_freq = int(0 * (nfft/2+1) / (fs/2.))
        # high_freq = int(15000 * (nfft/2+1) / (fs/2.))
        low_freq = 0
        high_freq = nfft/2 + 1

        # Applying low-pass filter

        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_lowpass_filter(data, cutOff, fs, order=4):
            b, a = butter_lowpass(cutOff, fs, order=order)
            y = lfilter(b, a, data)
            return y

        frames_array = butter_lowpass_filter(frames_array, cutOff=15000, fs=fs, order=10)

        # DWT

        # import pywt
        # c, d = pywt.dwt(frames_array, 'db1')

        # generating specgram
        Pxx, freqs, t, plot = pylab.specgram(
            frames_array,
            NFFT=nfft,
            Fs=fs,
            detrend=pylab.detrend_none,
            window=pylab.window_hanning,
            # sides='twosided',
            noverlap=int(nfft * 0.5),
            # pad_to=
            )

        # creating sequence of (feature, label)
        features_vectors, labels = [], []
        for n, interval in enumerate(intervals):
            start = interval._start_time / time_step
            end = interval._end_time / time_step
            features = Pxx[:, start:end]
            label = interval.text
            for i in xrange(np.shape(features)[1]):
                features_vectors.append(features[:, i])
            labels.append(alphabet[label])  # TODO shift this to switch targets: dense <--> sparse

        # if np.shape(feature_label[0])[1] != 128:
        #     print "short interval"
        return np.array(features_vectors), np.array(labels)

    def extract_mfcc_feature_vec(self, data, alphabet):
        n_ceps = 13
        features = np.zeros((1, n_ceps))
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
                rate = data.getFramerate()
                # ceps = mfcc(input=frames_in_interval, fs=rate, nceps=n_ceps)[0]
                ceps = pspf.mfcc(signal=np.array(frames_in_interval), samplerate=rate, numcep=n_ceps,
                                 highfreq=8000, appendEnergy=True)
                # , winlen=256./rate, winstep=160./rate)
                # print("frames in int", len(frames_in_interval))
                # print("ceps" , len(ceps))
            except Exception:
                print("ceps=0", data._textGridFile)
                raise

            for i in range(len(ceps)):
                #
                # FIXME why to normalize here???
                #
                # normalized_cep = preprocessing.normalize([ceps[i]])
                features = np.append(features, [ceps[i]], axis=0)
                feat_label_vec.append(alphabet[letter])  # TODO shift this to switch targets

        features = features[1:, :]
        # print("feat size", len(features))
        # print("lebel size", len(feat_label_vec))

        def compute_deltas(arrays):
            deltas = np.zeros(arrays.shape)
            for i in xrange(1, len(arrays) - 1):
                prev = arrays[i - 1]
                next = arrays[i + 1]
                deltas[i] = (next - prev) / 2
            # first delta
            this = arrays[0]
            next = arrays[1]
            deltas[0] = next - this
            # last delta
            this = arrays[len(arrays) - 1]
            prev = arrays[len(arrays) - 2]
            deltas[0] = this - prev
            return deltas

        # extending MFCCs with deltas
        features_deltas = compute_deltas(features)
        features_ddeltas = compute_deltas(features_deltas)

        def zero_mean(array):
            return (array - np.mean(array)) / (np.max(array) - np.min(array))

        # # normalizing
        # features = preprocessing.normalize(features)
        # features_deltas = preprocessing.normalize(features_deltas)
        # features_ddeltas = preprocessing.normalize(features_ddeltas)

        # # zero-mean normalizing
        # features = zero_mean(features)
        # features_deltas = zero_mean(features_deltas)
        # features_ddeltas = zero_mean(features_ddeltas)

        n = len(features)
        new_features = np.zeros((n, 3 * len(features[0])))
        for i in xrange(n):
            new_features[i] = np.append(features[i], [features_deltas[i], features_ddeltas[i]])

        return (new_features, np.array(feat_label_vec))

    def get_alphabet(self):
        return self.alphabet

    def get_mfcc_features(self):
        if self._mfcc_feature_label is None:
            feat_labs = []  # (MFCC, d, dd, label)
            self._mfcc_feature_label = []
            for data in self.parsed_files:
                self._mfcc_feature_label.append(self.extract_mfcc_feature_vec(data))

                # mfccs = [vec[0] for vec in feat_labs]
                # d = [vec[1] for vec in feat_labs]
                # dd = [vec[2] for vec in feat_labs]
                #
                # mfccs = preprocessing.normalize(mfccs)
                # d = preprocessing.normalize(d)
                # dd = preprocessing.normalize(dd)
                #
                # n = len(mfccs)
                # new_features = np.zeros((n, 3 * len(mfccs[0])))
                # for i in xrange(n):
                #     new_features[i] = np.append(mfccs[i], [d[i], dd[i]])
                #
                # self._mfcc_feature_label = [(feat, label) for feat, (_, label) in new_features, feat_labs]

        return self._mfcc_feature_label

    def get_spec_features(self, alphabet):
        if self.spec_feature_label is None:
            self.spec_feature_label = []
            for data in self.parsed_files:
                raw = self.extract_spectrogram_features(data, alphabet)
                # TODO what if  raw[0] - np.mean(raw[0])
                data = (preprocessing.normalize(raw[0]), raw[1])
                # data = ((raw[0] - np.mean(raw[0]))/(raw[0].max() - raw[0].min()), raw[1])
                self.spec_feature_label.append(data)

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
