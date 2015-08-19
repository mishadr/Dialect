__author__ = 'larisa'

import glob
import sys

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

    def _get_frame (self, wav_file):
        try:
            wav = wave.open(wav_file, mode="r")
            (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
            content = wav.readframes(nframes)
            samples = np.fromstring(content, dtype=types[sampwidth])
            mono_samples = samples[0::nchannels]

        except Exception:
            (mono_samples, framerate, sampwidth) = (0, 0, 0)

        return (mono_samples, framerate, sampwidth)

    def _check_textgrid_content(self, tiers):
        empty_marker = True
        for tier in tiers:
            if tier.text != "":
                empty_marker = False
                break
        return empty_marker

    def _intervalGetter (self, textGrid_file):
        if os.stat(textGrid_file).st_size == 0:
            return 'error'
        try:
            textgrid = tgt.io.read_textgrid(textGrid_file, encoding='utf-16')
            tiers = textgrid.tiers[0].intervals
            if self._check_textgrid_content(tiers):
                print("empty", textGrid_file)
                return 'error'
        except Exception:
            try:
                textgrid = tgt.io.read_textgrid(textGrid_file, encoding='utf-8')
                tiers = textgrid.tiers[0].intervals
                if self._check_textgrid_content(tiers):
                    return 'error'
            except Exception as e:
                tiers = 'error'
                print(e.reason)

        return tiers

    def __init__(self, wavFile, textGridFile):
        super(DataReader, self).__init__()
        (self._frames, self._framerate, self._sampwidth) = self._get_frame(wavFile)
        self._intervals = self._intervalGetter(textGridFile)
        self._wavFile = wavFile
        self._textGridFile = textGridFile

    def getFrames (self):
        return self._frames

    def getIntervals(self):
        return self._intervals

    def getFramerate(self):
        return self._framerate

    def getSampwidth(self):
        return self._sampwidth

    def getErr (self):
        if (self._framerate == self._sampwidth == 0):
            print(self._wavFile)
            return 1
        if (self._intervals == 'error'):
            return 1


        intervals = self.getIntervals()
        for i in range(len(self.getIntervals())):

            if (hasattr(intervals[i], '_end_time') == False or
                hasattr(intervals[i], '_start_time') == False or
                hasattr(intervals[i], 'text') == False):

                return 1

        return 0
