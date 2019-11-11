import os

import fragment
from conf import *

import utils

"""
Recompose an mp3 from fragments.
"""

def compose(song):
    title = song[:-4]
    song = None

    frags = fragment.from_directory(
        song=title,
        directory=GENERATED_DIR
    )

    for fr in frags:
        chunk = fr.get_wav_seg()

        if song is None:
            song = chunk
        else:
            song = song + chunk

            print(len(song))

    song.export("song.wav", format="wav")

def main():
    compose("Organ Fashioned Love Song v_0_1.mp3")

if __name__ == '__main__':
    main()
