
import helper.SpectralAdaptor as SA
import helper.SignalProcessor as SignalProcessor
import helper.DynamicAdaptor as DA



def doMastering(y_in, y_ref, sr, parameters):

    print("start Mastering!")
    y_in = SignalProcessor.preprocessSignal(y_in)
    y_ref = SignalProcessor.preprocessSignal(y_ref)

    print("preprocessed files!")

    y_in_chorus = SignalProcessor.getLoudestPart(y_in, sr, parameters)
    y_ref_chorus = SignalProcessor.getLoudestPart(y_ref, sr, parameters)

    print("extracted loudest parts!")

    y_filtered = SA.spectralAdaption(y_in, y_in_chorus, y_ref_chorus, parameters)
    #y_compressed = DA.dynamicAdaption(y_in, y_in_chorus, y_ref_chorus, parameters)

    print("adapted Spectrum!")

    return y_filtered