import os

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def onlyfiles(directory):
    return [f
            for f
            in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))]
