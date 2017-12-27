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

    return np.concatenate((y_compressed_left, y_compressed_right)).reshape((2, y_in.shape[1]))

def dynamicAdaptionDigitized(y_in, y_in_chorus, y_ref_chorus, parameters):


    values_in, counts_in = calCumulativeHistogramDigitized(y_in_chorus, parameters)
    values_ref, counts_ref = calCumulativeHistogramDigitized(y_ref_chorus, parameters)

    transferF = matchHistogramsDigitized(counts_in, values_ref, counts_ref)

    return y_in

def compressDigitized(y, values, transferF):
    y_compressed = np.zeros(y.shape)

    for i in range(values.size):
        y_compressed[np.equal(y, values[i])] = transferF[i]
        y_compressed[np.equal(-y, -values[i])] = -transferF[i]

    return y_compressed

def matchHistogramsDigitized(counts_in, values_ref, counts_ref):

    transferF = np.zeros(counts_in.size)
    for currentIndex in range(counts_in.size):
        matchedIndex = (np.abs(counts_ref - counts_in[currentIndex])).argmin()
        transferF[currentIndex] = values_ref[matchedIndex]

    return transferF



def limitSlope(transferF):

    F_diff = np.diff(transferF)

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
    y_abs = np.abs(SP.normalize(li.to_mono(y)))
    values, counts = np.unique(y_abs, return_counts = True)
    counts_cum = np.cumsum(counts)

    return values.astype("float32"), SP.normalize(counts_cum).astype("float32")


