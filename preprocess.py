import os

from pydub import AudioSegment

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/processed"

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
        print(songfile, i, len(c))

        chunk_path_out = os.path.join(OUTPUT_DIR,
                                      f"{songfile[:-4]}-{i}.wav")

        c.export(chunk_path_out, format="wav")

        i = i + 1
        
