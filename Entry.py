import controller.Mastering_Pipeline as mastering_pipeline
import helper.IO as io

path_in = "audios/LPP - Pisco (Remix)_short.wav"
path_ref = "audios/LPP - Como eh.mp3"
path_out = "audios/"


parameters = {'sampling_rate': 44100,
              'res_bits':16,
              "excerp_length_s":30 ,
              "n_fft": 2048,
              "kernel_length": 80,
              'max_transfer_slope': 2,
              'denoise_slope':2
              }
def main():


    y_in, sr = io.loadAudioFile(path_in, parameters['sampling_rate'])
    y_ref, sr = io.loadAudioFile(path_ref, parameters['sampling_rate'])

    y_out = mastering_pipeline.doMastering(y_in, y_ref, sr, parameters)

    io.writeOutAudioFile(path_out, "mastered5.wav", y_out, sr)


main()