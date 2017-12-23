import numpy as np
import librosa as li
import helper.SignalProcessor as SP


def dynamicAdaption(y_in, y_in_chorus, y_ref_chorus, parameters):

    #hist_in = calCumulativeHistogram(parameters, y_in_chorus)
    #hist_ref = calCumulativeHistogram(parameters, y_ref_chorus)

    counts_in, edges_in = calCumulativeHistogram(y_in_chorus, parameters)
    counts_ref, edges_ref = calCumulativeHistogram(y_ref_chorus, parameters)

    transferF = matchHistograms(counts_in, edges_ref, counts_ref)

    #limitSlope(transferF)

    y_compressed = compress(y_in, transferF)


    return y_compressed

def limitSlope(transferF):

    F_diff = np.diff(transferF)

def compress(y_in, transferF):
    y_compressed = np.zeos(y_in.shape)

    for i in range(transferF.size):
        x = 0

    return y_compressed


def matchHistograms(counts_in, edges_ref, counts_ref):

    transferF = np.ones(edges_ref.size)
    for currentIndex in range(counts_in.size):
        # Fix later: interpolate between the TWO nearest Edges!!
        if currentIndex == counts_in.size-1:
            print ("hallo")
        matchedIndex = (np.abs(counts_ref - counts_in[currentIndex])).argmin()
        transferF[currentIndex] = edges_ref[matchedIndex]
    return transferF


def calCumulativeHistogram(y, parameters):
    y_abs = np.abs(SP.normalize(li.to_mono(y)))
    counts, bin_edges = np.histogram(a=y_abs, bins=np.linspace(0,1,2**16+1, endpoint=True))
    counts_cum = np.cumsum(counts)

    return SP.normalize(counts_cum).astype("float32"), bin_edges.astype("float32")




