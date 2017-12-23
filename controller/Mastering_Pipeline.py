
import helper.SpectralAdaptor as SA
import helper.SignalProcessor as SP
import helper.DynamicAdaptor as DA



def doMastering(y_in, y_ref, sr, parameters):

    print("Start Mastering Process!")
    print("Start: Preprocessing Audio")
    y_in = SP.preprocessSignal(y_in)
    y_ref = SP.preprocessSignal(y_ref)

    print("Start: Extracting Chorus")

    y_in_chorus, y_in_start, y_in_end = SP.getLoudestPart(y_in, sr, parameters)
    y_ref_chorus, y_ref_start, y_ref_end = SP.getLoudestPart(y_ref, sr, parameters)

    print("Start: Equalizing")
    y_in_filtered = SA.spectralAdaption(y_in, y_in_chorus, y_ref_chorus, parameters)


    print("Start: Compressing")
    y_in_chorus_filtered = SP.updateChorusPart(parameters['kernel_length'], y_in_filtered, y_in_start, y_in_end )
    y_compressed = DA.dynamicAdaption(y_in_filtered, y_in_chorus_filtered, y_ref_chorus, parameters)

    print("adapted Spectrum!")

    return SP.normalize(y_compressed)


