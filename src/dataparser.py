__author__ = 'larisa'

import wave
import numpy as np
import tgt
import os

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}


class DataReader(object):
    """ Data extracted from given .wav file (monochannel frames) and
    corresponding .textGrid file (marked time intervals)
    """

    def _extract_frames(self, wav_file):
        """Extracts mono-channel frames (2-byte samples) from audio file"""
        try:
            wav = wave.open(wav_file, mode="r")
            (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
            content = wav.readframes(nframes)
            samples = np.fromstring(content, dtype=types[sampwidth])
            mono_samples = samples[0::nchannels]

        except Exception:
            (mono_samples, framerate, sampwidth) = (0, 0, 0)

        return mono_samples, framerate, sampwidth

    def _is_empty(self, tier):
        """ Checks whether all intervals are marked as ""
        """
        for int in tier:
            if int.text != "":
                return False
        return True

    def _extract_intervals(self, textGrid_file):
        """ Gets intervals from textGrid file or returns 'error' in case of error
        """
        if os.stat(textGrid_file).st_size == 0:
            return 'error'
        try:
            textgrid = tgt.io.read_textgrid(textGrid_file, encoding='utf-16')
            intervals = textgrid.tiers[0].intervals
            if self._is_empty(intervals):
                print("empty", textGrid_file)
                return 'error'
        except Exception:
            try:
                textgrid = tgt.io.read_textgrid(textGrid_file, encoding='utf-8')
                intervals = textgrid.tiers[0].intervals
                if self._is_empty(intervals):
                    return 'error'
            except Exception as e:
                intervals = 'error'
                print(e)

        return intervals

    def __init__(self, wavFile, textGridFile):
        super(DataReader, self).__init__()
        self._frames, self._framerate, self._sampwidth = self._extract_frames(wavFile)
        self._intervals = self._extract_intervals(textGridFile)
        self._wavFile = wavFile
        self._textGridFile = textGridFile

    def getFrames(self):
        return self._frames

    def getIntervals(self):
        return self._intervals

    def getFramerate(self):
        return self._framerate

    def getSampwidth(self):
        return self._sampwidth

    def is_correct(self):
        if self._framerate == self._sampwidth == 0:
            print "incorrect: " + str(self._wavFile)
            return False
        if self._intervals == 'error':
            return False

        for interval in self.getIntervals():
            if (hasattr(interval, '_end_time') == False or
                        hasattr(interval, '_start_time') == False or
                        hasattr(interval, 'text') == False):
                return False

        return True
