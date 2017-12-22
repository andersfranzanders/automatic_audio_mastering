
import helper.SpectralAdaptor as Spectral
import helper.SignalProcessor as SignalProcessor



def doMastering(y_in, y_ref, sr, parameters):

    print("start Mastering!")
    y_in = SignalProcessor.preprocessSignal(y_in)
    y_ref = SignalProcessor.preprocessSignal(y_ref)

    print("preprocessed files!")

    y_in_refrain = SignalProcessor.getLoudestPart(y_in, sr, parameters)
    y_ref_refrain = SignalProcessor.getLoudestPart(y_ref, sr, parameters)

    print("extracted loudest parts!")

    y_filtered = Spectral.spectralAdaption(y_in, y_in_refrain, y_ref_refrain, parameters)

    print("adapted Spectrum!")

    return y_filtered