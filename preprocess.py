import os

from conf import *
import librosa
import numpy as np
from pydub import AudioSegment
import utils

"""
This script takes a directory of MP3's (INPUT_DIR),
splits each into small chunks, then transforms
each chunk by a short-time Fourier transform to turn
each chunk into a numerical array.

These arrays are saved to an output directory.
"""

def chopchop():
    for songfile in utils.onlyfiles(INPUT_DIR):
        song_path_in = os.path.join(INPUT_DIR, songfile)
        song = AudioSegment.from_mp3(song_path_in)

        i = 0
        for c in utils.chunks(song,segment_duration):
            if len(c) < segment_duration:
                continue

            print(f"Processing {songfile} chunk {i}")
            chunk_path_out = os.path.join(OUTPUT_DIR,
                                          f"{songfile[:-4]}---{i}")

            # Write a small .wav file because librosa can't
            # handle big ones ?
            c.export(f"{chunk_path_out}.wav", format="wav")

            x, sr = librosa.load(f"{chunk_path_out}.wav")
            print(f"Sample rate: {sr}")
            assert sr == wav_sample_rate

            X = librosa.stft(x)
            Xdb = librosa.amplitude_to_db(X)

            print(f"Numpy array: Shape: {Xdb.shape}; Max: {Xdb.max()}; Min: {Xdb.min()}")
            np.save(chunk_path_out, Xdb)

            os.remove(f"{chunk_path_out}.wav")

            i = i + 1

def main():
    chopchop()

if __name__ == '__main__':
    main()
