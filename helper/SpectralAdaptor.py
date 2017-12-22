
import scipy
import librosa
import numpy as np
import helper.SignalProcessor as sp



def calAverageSpectrum(y, n_fft):
    STFT_left = calMagnitudeSTFT(y[0][:], n_fft)
    STFT_right = calMagnitudeSTFT(y[1][:], n_fft)

    Y_left_avg = np.average(abs(STFT_left), axis=1)
    Y_right_avg = np.average(abs(STFT_right), axis=1)

    return ( Y_left_avg + Y_right_avg ) / 2.0



def calMagnitudeSTFT(y_mono, n_fft):
    return librosa.stft(y=y_mono, n_fft=n_fft, hop_length=int(n_fft/2), win_length=n_fft, window=scipy.signal.hamming(n_fft, sym=False), center=False)
    #return np.abs(librosa.stft(y=y_mono, n_fft=n_fft, hop_length=n_fft, win_length=n_fft, window='boxcar', center=False))

def spectralImpulseToFilterKernel(H_raw, kernel_length):
     H_raw_cart = pol2cart(H_raw, np.zeros(H_raw.size))
     h_raw_comp = np.fft.ifft(H_raw_cart)
     h_raw_real = h_raw_comp.real
     h_shifted = np.roll(h_raw_real, int(kernel_length/2))
     h_truncated = h_shifted[:kernel_length]
     h_windowed = h_truncated * scipy.signal.hamming(kernel_length, sym=False)

     return h_windowed


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    comp = x[:] + 1j * y[:]
    comp_rev = np.copy(comp[::-1])
    whole = np.concatenate((comp[:], comp_rev[1:-1]),axis=0)

    return whole



def spectralAdaption(y_in, y_in_refrain, y_ref_refrain, parameters):

    Y_avg_in = calAverageSpectrum(y_in_refrain, parameters["n_fft"])
    Y_avg_ref = calAverageSpectrum(y_ref_refrain , parameters["n_fft"])
    Y_diff = Y_avg_ref / Y_avg_in

    h = spectralImpulseToFilterKernel(Y_diff, parameters["kernel_length"])

    y_filtered_left = np.convolve(y_in[0][:], h)
    y_filtered_right = np.convolve(y_in[1][:], h)
    y_filtered = np.concatenate(([y_filtered_left], [y_filtered_right]), axis=0)

    y_filtered = sp.normalize(y_filtered)

    return y_filtered

