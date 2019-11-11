from conf import *
import os
import librosa
import numpy as np
from pydub import AudioSegment

from pydub.playback import play
import utils

class Fragment():
    song = None
    number = None

    mp3_seg = None
    wav_seg = None
    np_data = None

    def __init__(self,
                 song,
                 number,
                 mp3_seg = None,
                 wav_seg = None,
                 np_data = None
    ):
        self.song = song
        self.number = number
        self.mp3_seg = mp3_seg
        self.wav_seg = wav_seg
        self.np_data = np_data

    def path(self,dest="training"):
        directory = None

        if dest == "training":
            directory = TRAINING_FRAGMENT_DIR
        if dest == "generated":
            directory = GENERATED_DIR

        return os.path.join(directory,
                            f"{self.song}---{self.number}")

    def mp3_to_np(self, save=True):
        # Write a small .wav file because librosa can't
        # handle big ones ?
        self.mp3_seg.export(f"{self.path()}.wav",
                            format="wav")
        
        x, sr = librosa.load(f"{self.path()}.wav")
        print(f"Sample rate: {sr}")
        assert sr == wav_sample_rate

        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(X)

        self.np_data = Xdb

        os.remove(f"{self.path()}.wav")
        
        if save:
            print(f"Numpy array: Shape: {Xdb.shape}; Max: {Xdb.max()}; Min: {Xdb.min()}")
            np.save(self.path(), self.np_data)

    def np_to_wav(self, save = False, dest="generated"):
        wav_path = self.path(dest=dest) + ".wav"

        Xdb = self.np_data
        X2 = librosa.db_to_amplitude(Xdb)
        x2 = librosa.core.istft(X2)

        librosa.output.write_wav(wav_path,
                                 x2,
                                 wav_sample_rate)
        chunk = AudioSegment.from_wav(wav_path)

        if not save:
            os.remove(wav_path)

        self.wav_seg = chunk

        return self.wav_seg

    def get_wav_seg(self):
        if self.wav_seg:
            return self.wav_seg
        elif self.np_data:
            return self.np_to_wav()
        else:
            return None

    def load_np_data(self):
        self.np_data = np.load(self.path() + ".npy")


def from_directory(song=None, directory = TRAINING_FRAGMENT_DIR):

    filenames = utils.onlyfiles(directory)

    fns = [(fi,
            fi.split("---")[0],
            int(fi.split("---")[1][:-4]),
            fi.split("---")[1][-4:])
           for
           fi
           in
           filenames
           if
           "---" in fi
    ]

    fns = sorted(fns, key=lambda x: x[2])
    

    for fi,fi_song,fi_number,fi_type in fns:
        
        try:
            if song and fi_song != song:
                continue

            np_data = None
            if fi_type == ".npy":
                np_data = np.load(os.path.join(
                    directory,
                    fi),
                                allow_pickle=True)

            wav_seg = None
            if fi_type == ".wav":
                wav_seg = AudioSegment.from_wav(
                    os.path.join(
                        directory,
                        fi))

            frag = Fragment(fi_song,
                            fi_number,
                            np_data = np_data,
                            wav_seg = wav_seg
            )
            yield frag

        except Exception as e:
            print(e)
            pass

    
