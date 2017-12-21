

import librosa


target_sr = 44100


def loadAudioFile(filePath):
    y, sr = librosa.load(filePath, sr=target_sr, mono=False)

    return y, sr


def writeOutAudioFile(path, name, y, sr):
    librosa.output.write_wav(path+name, y, sr)