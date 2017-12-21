
import helper.Spectral as Spectral
import helper.SignalProcessor as SignalProcessor



def doMastering(y_in, y_ref, sr, parameters):



    y_in = SignalProcessor.preprocessSignal(y_in)
    y_ref = SignalProcessor.preprocessSignal(y_ref)

    y_filtered = Spectral.spectralAdaption(y_in, y_ref, parameters)


    return y_filtered