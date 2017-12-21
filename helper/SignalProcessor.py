import librosa
import numpy as np


def preprocessSignal(y):

    y_processed = librosa.util.normalize(y)

    if y.ndim != 2:
        y_stereo = np.zeros((2, y.size), dtype=y_processed.dtype)
        for i in range(2):
            y_stereo[i][:] = y_processed[:]

        return y_stereo

    return y_processed





