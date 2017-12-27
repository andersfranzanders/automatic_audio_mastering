

import librosa


def loadAudioFile(filePath, sampling_rate):
    y, sr = librosa.load(filePath, sr=sampling_rate, mono=False)

    return y, sr


def writeOutAudioFile(path, name, y, sr):
    librosa.output.write_wav(path+name, y, sr)