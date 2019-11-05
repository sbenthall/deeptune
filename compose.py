import os

from conf import *
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import utils

"""
Recompose an mp3 from fragments.
"""

def chunknumber(chunkname):
    return int(chunkname[:-4].split("---")[1])

def compose(song):
    title = song[:-4]
    
    chunkfiles = sorted([chunkfile
                         for
                         chunkfile
                         in utils.onlyfiles(OUTPUT_DIR)
                         if chunkfile.startswith(title)
                         and chunkfile.endswith(".npy")],
                        key=chunknumber
    )

    song = None
    
    for cf in chunkfiles:
        wav_file_name = cf[:-4]
        
        Xdb = np.load(os.path.join(OUTPUT_DIR,cf))
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
