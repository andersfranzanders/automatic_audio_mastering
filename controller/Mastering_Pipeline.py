
import helper.Spectral as Spectral




def doMastering(y_in, y_ref, sr, parameters):
    y_out = y_in

    Y_out_avg = Spectral.calAverageSpectrum(y_in, parameters["n_fft"])



    return y_out