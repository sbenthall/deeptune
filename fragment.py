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

    def path(self):
        return os.path.join(OUTPUT_DIR,
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

    def np_to_wav(self):
        wav_file_name = self.song

        Xdb = self.np_data
        X2 = librosa.db_to_amplitude(Xdb)
        x2 = librosa.core.istft(X2)

        librosa.output.write_wav(wav_file_name,
                                 x2,
                                 wav_sample_rate)
        chunk = AudioSegment.from_wav(wav_file_name)

        os.remove(wav_file_name)

        self.wav_seg = chunk

        return self.wav_seg


def from_directory(song=None):
    filenames = utils.onlyfiles(OUTPUT_DIR)

    for fi in filenames:
        fi_song, fi_number_type = fi.split("---")

        try:
            if song and fi_song != song:
                continue

            fi_number = int(fi_number_type[:-4])

            frag = Fragment(fi_song,
                            fi_number,
                            np_data = np.load(os.path.join(
                                OUTPUT_DIR,
                                fi),
                                              allow_pickle=True))
            yield frag

        except:
            pass

    
