import librosa
import numpy as np


def load_to_db(inp, duration = 30):
    x, sr = librosa.load(inp, duration = duration)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(X)
    return Xdb, sr


def mix_transform(Xdb1,Xdb2):
    db_range = Xdb1.shape[0]
    shared_length = min(Xdb1.shape[1],Xdb2.shape[1])
    
    mix = np.random.random(size=(db_range,shared_length)) > .5
    
    Xdb3 = mix * Xdb1[:,:shared_length] + np.logical_not(mix) * Xdb2[:,:shared_length]
    
    return Xdb3


def merge(in1, in2, out, transform_method = mix_transform, duration=30):
    Xdb1, sr = load_to_db(in1,duration = duration)
    Xdb2, sr = load_to_db(in2, duration = duration )
    
    Xdb3 = transform_method(Xdb1,Xdb2)
    X3 = librosa.db_to_amplitude(Xdb3)
    x3 = librosa.core.istft(X3)
    librosa.output.write_wav(out, x3, sr)


### Testing
input1_path = '../samples/audio/busta_rhymes_hits_for_days.mp3'
input2_path = '../samples/steve-morrell/Life Gift.mp3'
output_path = 'merge-test-out.wav'
merge(input1_path, input2_path, output_path, duration = 7)
