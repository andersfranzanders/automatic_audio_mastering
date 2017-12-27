import controller.Mastering_Pipeline as mastering_pipeline
import helper.IO as io

path_in = "audios/LPP - BST_short.wav"
path_ref = "audios/LPP - Como eh.mp3"
path_out = "audios/"


parameters = {'sampling_rate': 44100,
              'res_bits': 16,
              'excerp_length_s': 30,
              'n_fft': 2048,
              'kernel_length': 80,
              'max_transfer_slope': 2,
              'denoise_slope': 3
              }
def main():

    y_in, sr = io.loadAudioFile(path_in, parameters['sampling_rate'])
    y_ref, sr = io.loadAudioFile(path_ref, parameters['sampling_rate'])

    print("----------- Start Mastering -----------")
    print("Track to Master: " + path_in)
    print("Reference Track: " + path_ref)
    print("Parameters: ")
    print(parameters)

    print("----------- Progress: -----------")
    y_out = mastering_pipeline.doMastering(y_in, y_ref, sr, parameters)

    print("----------- Finished Mastering ---------")

    io.writeOutAudioFile(path_out, "mastered.wav", y_out, sr)


main()