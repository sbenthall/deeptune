import os

import fragment
from conf import *
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
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
        wav_file_name = fr.song
        
        Xdb = fr.np_data
        X2 = librosa.db_to_amplitude(Xdb)
        x2 = librosa.core.istft(X2)

        librosa.output.write_wav(wav_file_name,
                                 x2,
                                 wav_sample_rate)
        chunk = AudioSegment.from_wav(wav_file_name)

        if song is None:
            song = chunk
        else:
            song = song + chunk

        os.remove(wav_file_name)
        print(len(song))

    #play(song)

    song.export("song.wav", format="wav")

def main():
    compose("11 - An Old Fashioned Love Song (Single Version).mp3")

if __name__ == '__main__':
    main()
