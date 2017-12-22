import controller.Mastering_Pipeline as mastering_pipeline
import helper.IO as io

path_in = "audios/LPP - MGA.wav"
path_ref = "audios/LPP - Como eh.mp3"
path_out = "audios/"


parameters = {"n_fft": 2048, "kernel_length": 80, "excerp_length_s":30 , 'res_bits':16 }



def main():


    y_in, sr = io.loadAudioFile(path_in)
    y_ref, sr = io.loadAudioFile(path_ref)

    y_out = mastering_pipeline.doMastering(y_in, y_ref, sr, parameters)

    io.writeOutAudioFile(path_out, "mastered.wav", y_out, sr)


main()