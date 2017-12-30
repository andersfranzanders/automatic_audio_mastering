import numpy as np
import librosa as li
import helper.SignalProcessor as SP
from datetime import datetime


def adaptDynamics(y_in, y_in_chorus, y_ref_chorus, parameters):


    values_in, counts_in = calCumulativeHistogram(y_in_chorus, parameters)
    values_ref, counts_ref = calCumulativeHistogram(y_ref_chorus, parameters)

    print("Calculating Transferfunction")
    transferF = matchHistograms(counts_in, values_ref, counts_ref)

    transferF_slimited = limitSlopeOfTransferF(transferF, values_in, parameters['max_transfer_slope'], parameters['res_bits'])
    transferF_denoised = denoiseTransferF(transferF_slimited, values_in, parameters['denoise_slope'])

    print("Applying Transfer Function")
    y_compressed = applyCompressionPerformiced(y_in, values_in, transferF_denoised, parameters)

    return y_compressed

def applyCompressionPerformiced(y, values_in, transferF, parameters):

    y_digitized, _ = SP.digitizeAmplitudesStereo(y, parameters['res_bits'])
    y_left = y_digitized[0,:]
    y_right = y_digitized[1,:]

    iterator = (applyTransferIterator(i, values_in, transferF) for i in y_left)
    y_left_out = np.fromiter(iterator, np.float, count = y_left.size)

    iterator = (applyTransferIterator(i, values_in, transferF) for i in y_right)
    y_right_out = np.fromiter(iterator, np.float, count = y_right.size)

    return np.concatenate((y_left_out, y_right_out)).reshape(y.shape)

def applyTransferIterator(i, values, transferF):
    transferValue = transferF[np.equal(abs(i), values).argmax()]
    if i < 0:
        return - transferValue
    return transferValue


def denoiseTransferF(transferF, values_in, slope):
    straightLine = slope * values_in
    #straightLine = (slope+1)*values_in
    mask = np.greater(transferF, straightLine)
    transferF[mask] = straightLine[mask]

    return transferF

def applyCompression_old(y, values, transferF, parameters):
    y_digitized, _ = SP.digitizeAmplitudesStereo(y, parameters['res_bits'])
    y_compressed = np.zeros(y.shape)
    y_values_pos = np.unique(np.abs(y_digitized))

    for i in y_values_pos:
        transferValue = transferF[(i == values).argmax()]
        y_compressed[y_digitized == i] = transferValue
        y_compressed[y_digitized == -i] = -transferValue
        # np.copyTo()

    return y_compressed

def matchHistograms(counts_in, values_ref, counts_ref):

    transferF = np.zeros(counts_in.size)
    for currentIndex in range(counts_in.size):
        matchedIndex = (np.abs(counts_ref - counts_in[currentIndex])).argmin()
        transferF[currentIndex] = values_ref[matchedIndex]

    return transferF



def limitSlopeOfTransferF(transferF, values_in, max_slope_faktor, res_bits):

    transferF_pre = np.concatenate((np.asarray([0]), transferF))

    max_slope = max_slope_faktor *(values_in[1] - values_in[0])
    F_diff = np.diff(transferF_pre)
    SP.limit(F_diff, max_slope)

    F_slimited = np.cumsum(F_diff)

    ### Mastering by adding difference
    #F_slimited_norm_dig = F_slimited + (1 - F_slimited.max())

    ### Mastering by normalizing and redigitalizing
    F_slimited_norm = SP.normalize(F_slimited)
    F_slimited_norm_dig, _ = SP.digitizeAmplitudesMono(F_slimited_norm, res_bits)


    return F_slimited_norm_dig

def calCumulativeHistogram(y, parameters):
    #y_abs = np.abs(SP.digitizeAmplitudes(li.to_mono(y)))
    y_abs = SP.normalize(np.abs(li.to_mono(y)))
    y_abs_dig, values_all = SP.digitizeAmplitudesMono(y_abs, parameters['res_bits'])
    values_all_pos = values_all[2**(parameters['res_bits'] - 1) :]

    values_y, counts_y = np.unique(y_abs_dig, return_counts = True)
    counts_all = np.zeros(values_all_pos.size )
    counts_all[np.in1d(values_all_pos, values_y)] = counts_y

    counts_cum = np.cumsum(counts_all)

    return values_all_pos, SP.normalize(counts_cum)


