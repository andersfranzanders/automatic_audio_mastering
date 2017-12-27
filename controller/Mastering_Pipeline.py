
import helper.SpectralAdaptor as SA
import helper.SignalProcessor as SP
import helper.DynamicAdaptor as DA




def doMastering(y_in, y_ref, sr, parameters):

    print("Start: Preprocessing Audio")
    y_in = SP.preprocessSignal(y_in, parameters)
    y_ref = SP.preprocessSignal(y_ref, parameters)

    print("Start: Searching for representative Parts")

    y_in_chorus, y_in_start, y_in_end = SP.getLoudestPart(y_in, sr, parameters)
    y_ref_chorus, y_ref_start, y_ref_end = SP.getLoudestPart(y_ref, sr, parameters)

    print("Start: Equalizing Stage ")
    y_in_filtered = SA.spectralAdaption(y_in, y_in_chorus, y_ref_chorus, parameters)
    y_in_chorus_filtered = SP.updateChorusPart(parameters['kernel_length'], y_in_filtered, y_in_start, y_in_end )

    print("Start: Compressing Stage")
    y_compressed = DA.adaptDynamics(y_in_filtered, y_in_chorus_filtered, y_ref_chorus, parameters)

    return SP.normalize(y_compressed)


