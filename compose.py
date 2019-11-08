import os

import fragment
from conf import *

import utils

"""
Recompose an mp3 from fragments.
"""

def compose(song):
    title = song[:-4]
    
    frags = sorted(fragment.from_directory(song=title),
                        key=lambda x: x.number
    )

    song = None
    
    for fr in frags:
        chunk = fr.np_to_wav()

        if song is None:
            song = chunk
        else:
            song = song + chunk

        print(len(song))

    #play(song)

    song.export("song.wav", format="wav")

def main():
    compose("11 - An Old Fashioned Love Song (Single Version).mp3")

if __name__ == '__main__':
    main()
