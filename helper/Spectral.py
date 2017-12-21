
import scipy
import librosa
import numpy as np



def calAverageSpectrum(y, n_fft):

    Y_left = calSTFT(y[0][:], n_fft)
    Y_right = calSTFT(y[1][:], n_fft)

    Y_left_avg = np.average(abs(Y_left), axis=0)
    Y_right_avg = np.average(abs(Y_right), axis=0)

    return Y_left_avg



def calSTFT(y_mono, n_fft):
    return librosa.stft(y=y_mono, n_fft=n_fft, hop_length=int(n_fft/2), win_length=n_fft, window=scipy.signal.hamming(n_fft, sym=False), center=False)