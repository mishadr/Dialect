__author__ = 'larisa'
# ecoding: utf8
import wave
import numpy as np
import tgt
import os

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}


def extract_frames(wav_file):
    """Extracts mono-channel frames (2-byte samples) from audio file"""
    try:
        wav = wave.open(wav_file, mode="r")
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
        content = wav.readframes(nframes)
        wav.close()
        if sampwidth == 1:
            print "monochannel: " + wav_file
        samples = np.fromstring(content, dtype=types[sampwidth])
        mono_samples = samples[0::nchannels]

    except Exception:
        (mono_samples, framerate, sampwidth) = ([], 1, 0)

    return mono_samples, framerate, sampwidth


class DataReader(object):
    """ Data extracted from given .wav file (monochannel frames) and
    corresponding .textGrid file (marked time intervals)
    """

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
                print(e)
                return 'error'

        # exclude rare symbols
        exclude = [
            # 1 time:
            u"ɨ:", u"tt", u"[э]", u"p˙", u"(15г)", u"җ'", u"tɕ", u"tt\^j", u"eˁ", u"oa", u"cʃ", u"(придыхание)",
            u"т-т", u"(21г)", u"нн", u"ъ(о)", u"b\^j", u"призв.", u"(29б)", u"ç", u"(50б)", u"ā", u"сс", u"кк",
            u"(9д)", u"ea", u"(5е)", u"n(?)", u"х'", u"[БАА: 4]", u"(12г)", u"(4и)", u"\aeu", u"dʒ", u"d’҇z’",
            u"d\^j", u"uj", u"\ri", u"\ct\:f", u"oj", u"ʃ'", u"(смычка d)", u"aˤ", u"yˁ:", u"(11к)", u"b'",
            u"(призв.)", u"\aeo", u"[ВЗТ]", u"ʥ", u"\o-u", u"éo", u"\\u-\:f", u"\cxx", u"ЛЗС", u"\tf", u"(24б)",
            u"jа[иа]", u"(25д)", u"дж'", u"Z", u"e\:f", u"ɦ", u"ue", u"ua", u"əó", u"io", u"ó:", u"P", u"е",
            u"(l\^j)", u"(глух. пр.)", u"(22в)", u"(19в)", u"(ЛАЛ)", u"кап", u"ӧ", u"ts\^j", u"ə:", u"(э)",
            u"[БАА: 3]", u"(1г)",
            # 2 times:
            u"øˤ", u"eí", u"\h^\^w", u"[БАА: 1]", u"\sh", u"jɔ:", u"β", u"eó", u"o\:f", u"[БАА: 2]", u"ɔi",
            u"áj", u"w\ff", u"χ (γ)", u"uo", u"б'", u"\o-\:f", u"\nj", u"aj", u"f'", u"úi", u"új", u"δ", u"с",
            u"óe", u"uˁ", u"aˁ"]
        for int in intervals:
            if int.text in exclude:
                return 'error'

        # checking whether last interval ends further than frames end
        interval_end = intervals[-1]._end_time
        frame_end = 1.0 * len(self._frames) / self._framerate
        if interval_end > frame_end + 0.0001:
            return 'error'

        return intervals

    def __init__(self, wavFile, textGridFile):
        super(DataReader, self).__init__()
        self._frames, self._framerate, self._sampwidth = extract_frames(wavFile)
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
