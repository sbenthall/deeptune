import os

from fragment import Fragment
from conf import *

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
    for songfile in utils.onlyfiles(RAW_MP3_DIR):
        song_path_in = os.path.join(RAW_MP3_DIR, songfile)
        song = AudioSegment.from_mp3(song_path_in)

        for i, c in enumerate(utils.chunks(song,segment_duration)):
            if len(c) < segment_duration:
                continue

            print(f"Processing {songfile} chunk {i}")
            fragment = Fragment(f"{songfile[:-4]}",
                                i,
                                mp3_seg = c
            )

            fragment.mp3_to_np()

def main():
    chopchop()

if __name__ == '__main__':
    main()
