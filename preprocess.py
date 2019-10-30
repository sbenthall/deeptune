import os

from pydub import AudioSegment

DATA_DIR = "data/input"

segment_duration = 1000

onlyfiles = [os.path.join(DATA_DIR, f)
             for f
             in os.listdir(DATA_DIR)
             if os.path.isfile(os.path.join(DATA_DIR, f))]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

for songfile in onlyfiles:
    song = AudioSegment.from_mp3(songfile)

    i = 0
    for c in chunks(song,segment_duration):
        print(songfile, i, len(c))
        i = i  + 1
        
