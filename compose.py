import os

import fragment
from conf import *

import utils

"""
Recompose an mp3 from fragments.
"""

def compose(song, directory=GENERATED_DIR):
    title = song
    song = None

    frags = fragment.from_directory(
        song=title,
        directory=directory
    )

    for fr in frags:
        chunk = fr.get_wav_seg()

        if song is None:
            song = chunk
        else:
            song = song + chunk

            print(len(song))

    song.export(f"{title}.wav", format="wav")

