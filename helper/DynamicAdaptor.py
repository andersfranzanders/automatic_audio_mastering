import numpy as np
import librosa as li
import helper.SignalProcessor as SP


def dynamicAdaption(y_in, y_in_chorus, y_ref_chorus, parameters):

    #hist_in = calCumulativeHistogram(parameters, y_in_chorus)
    #hist_ref = calCumulativeHistogram(parameters, y_ref_chorus)


    counts_in, edges_in = calCumulativeHistogram(y_in_chorus, parameters)
    counts_ref, edges_ref = calCumulativeHistogram(y_ref_chorus, parameters)

    print("Start: Buildings Transfer-Function!")
    transferF = matchHistograms(counts_in, edges_ref, counts_ref)

    #limitSlope(transferF)

    print("Start: Compressing!")
    y_compressed_left = compress(y_in[0], edges_in, transferF)
    y_compressed_right = compress(y_in[1], edges_in, transferF)

    return np.concatenate((y_compressed_left, y_compressed_right)).reshape((1, y_in.shape[1]))

def dynamicAdaptionDigitized(y_in, y_in_chorus, y_ref_chorus, parameters):


    values_in, counts_in = calCumulativeHistogramDigitized(y_in_chorus, parameters)
    values_ref, counts_ref = calCumulativeHistogramDigitized(y_ref_chorus, parameters)

    print("Start: Buildings Transfer-Function!")
    transferF = matchHistogramsDigitized(counts_in, values_ref, counts_ref)

    transferF_slimited = limitSlope(transferF, values_in, parameters['max_transfer_slope'])
    transferF_denoised = denoiseTransferF(transferF_slimited, values_in, parameters['max_transfer_slope'])

    print("Start: Compressing!!")
    y_compressed = compressDigitized(y_in, values_in, transferF_denoised, parameters)

    #y_check = compressDigitized(y_in_chorus, values_in, transferF, parameters)
    #values_check, counts_check = calCumulativeHistogramDigitized(y_check, parameters)

    return y_compressed

def denoiseTransferF(transferF, values_in, slope):
    straightLine = (slope+1)*values_in
    mask = np.greater(transferF, straightLine)
    transferF[mask] = straightLine[mask]

    return transferF

def compressDigitized(y, values, transferF, parameters):
    y_digitized, _ = SP.digitizeAmplitudesStereoPlus1(y, parameters['res_bits'])
    y_compressed = np.zeros(y.shape)
    y_values_pos = np.unique(np.abs(y_digitized))

    for i in y_values_pos:
        transferValue = transferF[np.equal(i,values).argmax()]
        y_compressed[np.equal(y_digitized, i)] = transferValue
        y_compressed[np.equal(y_digitized, -i)] = -transferValue

    return y_compressed

def matchHistogramsDigitized(counts_in, values_ref, counts_ref):

    transferF = np.zeros(counts_in.size)
    for currentIndex in range(counts_in.size):
        matchedIndex = (np.abs(counts_ref - counts_in[currentIndex])).argmin()
        transferF[currentIndex] = values_ref[matchedIndex]

    return transferF



def limitSlope(transferF, values_in, max_slope_faktor):

    transferF_pre = np.concatenate((np.asarray([0]), transferF))

    max_slope = max_slope_faktor *(values_in[1] - values_in[0])
    F_diff = np.diff(transferF_pre)
    SP.limit(F_diff, max_slope)

    F_slimited = np.cumsum(F_diff)
    F_slimited = F_slimited + (1 - F_slimited.max())

    return F_slimited

## Bits = 10 :  F.diff.max() = 0.005
## Bits = 9: F_diff.max() = 0.011
## Bits = 8 : F_diff.max() = 0.01562
## Bits = 4 : F_diff.max() = 0.25


def compress(y, edges, transferF):
    y_compressed = np.zeros(y.shape)

    for i in range(edges.size-1):
        if i == (edges.size-1):
            mask_positive = np.greater_equal(y, edges[i]) & np.less_equal(y, edges[i + 1])
            mask_negative = np.less_equal(y, -edges[i]) & np.greater_equal(y, -edges[i + 1])
        else:
            mask_positive = np.greater_equal(y, edges[i]) & np.less(y, edges[i+1])
            mask_negative = np.less_equal(y, -edges[i]) & np.greater(y, -edges[i + 1])
        y_compressed[mask_positive] = transferF[i]
        y_compressed[mask_negative] = -transferF[i]

    return y_compressed


def matchHistograms(counts_in, edges_ref, counts_ref):

    transferF = np.ones(edges_ref.size)
    for currentIndex in range(counts_in.size):
        # Fix later: interpolate between the TWO nearest Edges!!
        matchedIndex = (np.abs(counts_ref - counts_in[currentIndex])).argmin()
        transferF[currentIndex] = edges_ref[matchedIndex]

    transferF[-1] = transferF[-2]
    return transferF


def calCumulativeHistogram(y, parameters):
    y_abs = np.abs(SP.normalize(li.to_mono(y)))
    counts, bin_edges = np.histogram(a=y_abs, bins=np.linspace(0,1,2**16+1, endpoint=True))
    counts_cum = np.cumsum(counts)

    return SP.normalize(counts_cum).astype("float32"), bin_edges.astype("float32")

def calCumulativeHistogramDigitized(y, parameters):
    #y_abs = np.abs(SP.digitizeAmplitudes(li.to_mono(y)))
    y_abs = SP.normalize(np.abs(li.to_mono(y)))
    y_abs_dig, values_all = SP.digitizeAmplitudesMonoPlus1(y_abs, parameters['res_bits'])
    values_all_pos = values_all[2**(parameters['res_bits'] - 1) :]

    values_y, counts_y = np.unique(y_abs_dig, return_counts = True)
    counts_all = np.zeros(values_all_pos.size )
    counts_all[np.in1d(values_all_pos, values_y)] = counts_y

    counts_cum = np.cumsum(counts_all)

    return values_all_pos, SP.normalize(counts_cum)


