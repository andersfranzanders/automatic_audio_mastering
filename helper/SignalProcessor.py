import librosa
import numpy as np


def preprocessSignal(y, parameters):

    # y_processed = librosa.util.normalize(y)
    y_processed = normalize(y)

    if y.ndim != 2:
        y_stereo = np.zeros((2, y.size), dtype=y_processed.dtype)
        for i in range(2):
            y_stereo[i][:] = y_processed[:]

        return digitizeAmplitudes(y_stereo, parameters['res_bits'])

    return digitizeAmplitudes(y_processed, parameters['res_bits'])


def normalize(y):
    return y / np.max(np.abs(y))

def getLoudestPart(y, sr, parameters):

    y_squared = librosa.to_mono(y)**2

    numSamples = sr * parameters['excerp_length_s']
    start = 0
    end = y_squared.size + 1

    if numSamples < y_squared.size:
        rmsValues = np.zeros(y_squared.size - numSamples + 1)
        rmsValues[0] = sum(y_squared[:numSamples])
        for i in range(1, y_squared.size - numSamples + 1):
            rmsValues[i] = rmsValues[i-1] - y_squared[i-1] + y_squared[numSamples+i-1]
        start = np.argmax(rmsValues)
        end = start+numSamples

    return y[:, start:end], start, end

def calRMS(y):
    return np.sqrt(np.mean(y**2))


def calSquaredSum(y):
    return np.sum(y**2)

def updateChorusPart(kernelLength, y_filtered, y_start, y_end):
    offsetByFiltering = kernelLength - 1
    return y_filtered[:, y_start + offsetByFiltering:y_end + offsetByFiltering]

def digitizeAmplitudes(y, bitdepth):

    bins = np.linspace(-1, 1 + 2/(2 ** bitdepth - 1), 2 ** bitdepth + 1)
    y_left = bins[np.digitize(y[0,:],bins) - 1]
    y_right = bins[np.digitize(y[1,:],bins) - 1]

    #bins = np.linspace(-1, 1, 2**bitdepth + 1)
    #y_left = ( np.digitize(y[0, :], bins) - 1) / (2**(bitdepth-1) ) - 1
    #y_right= ( np.digitize(y[1, :], bins) - 1) / (2**(bitdepth-1) ) - 1
    return np.concatenate((y_left, y_right)).reshape(y.shape)