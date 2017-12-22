import numpy as np
import librosa as li
import helper.SignalProcessor as SignalProcessor


def dynamicAdaption(y_in, y_in_chorus, y_ref_chorus, parameters):

    #hist_in = calCumulativeHistogram(parameters, y_in_chorus)
    #hist_ref = calCumulativeHistogram(parameters, y_ref_chorus)

    h_in_values, h_in_counts = calCumulativeHistogram(y_in_chorus)


    #matchHistograms(hist_in, hist_ref)

    return 0




def matchHistograms(hist_in, hist_ref):



    return 0


def calCumulativeHistogram(y):

    h_values, h_counts = np.unique(np.abs(y), return_counts=True)
    h_cum_counts = np.zeros(h_counts.size)
    h_cum_counts[:] = [ np.sum(h_counts[:i + 1]) for i in range(h_counts.size) ]
    return h_values, h_cum_counts




# def calCumulativeHistogram(parameters, y):
#
#     y_mono = SignalProcessor.normalize(li.to_mono(y))
#
#     hist, bin_edges = np.histogram(a=y, bins=2 ** parameters['res_bits'], density=True)
#     hist_cumulative = np.zeros(hist.size)
#     for i in range(hist_cumulative.size):
#         hist_cumulative[i] = np.sum(hist[:i + 1])
#     return hist_cumulative / np.max(hist_cumulative)



