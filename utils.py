import os

def onlyfiles(directory):
    return [f
            for f
            in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))]
