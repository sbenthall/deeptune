import os

import librosa
import numpy as np
from pydub import AudioSegment

"""
This script takes a directory of MP3's (INPUT_DIR),
splits each into small chunks, then transforms
each chunk by a short-time Fourier transform to turn
each chunk into a numerical array.

These arrays are saved to an output directory.
"""

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/processed"

#number of miliseconds
segment_duration = 1000

onlyfiles = [f
             for f
             in os.listdir(INPUT_DIR)
             if os.path.isfile(os.path.join(INPUT_DIR, f))]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

for songfile in onlyfiles:
    song_path_in = os.path.join(INPUT_DIR, songfile)
    song = AudioSegment.from_mp3(song_path_in)

    i = 0
    for c in chunks(song,segment_duration):
        if len(c) < segment_duration:
            continue

        print(f"Writing DB array for {songfile} chunk {i}")
        chunk_path_out = os.path.join(OUTPUT_DIR,
                                      f"{songfile[:-4]}---{i}")

        # Write a small .wav file because librosa can't
        # handle big ones ?
        c.export(f"{chunk_path_out}.wav", format="wav")

        x, sr = librosa.load(f"{chunk_path_out}.wav")
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(X)

        np.save(chunk_path_out, Xdb)

        os.remove(f"{chunk_path_out}.wav")

        i = i + 1
        
