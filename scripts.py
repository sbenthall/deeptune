import librosa
import numpy as np


def load_to_amp(inp, duration = 30):
    x, sr = librosa.load(inp, duration = duration)
    X = librosa.stft(x)
    return X, sr


def mix_transform(X1,X2):
    amp_range = X1.shape[0]
    shared_length = min(X1.shape[1],X2.shape[1])
    
    mix = np.random.random(size=(amp_range,shared_length)) > .5
    
    X3 = mix * X1[:,:shared_length] + np.logical_not(mix) * X2[:,:shared_length]
    
    return X3


def merge(in1, in2, out, transform_method = mix_transform, duration=30):
    X1, sr = load_to_amp(in1,duration = duration)
    X2, sr = load_to_amp(in2, duration = duration )
    
    X3 = transform_method(X1,X2)
    x3 = librosa.core.istft(X3)
    librosa.output.write_wav(out, x3, sr)


### Testing
input1_path = '../samples/audio/busta_rhymes_hits_for_days.mp3'
input2_path = '../samples/steve-morrell/Life Gift.mp3'
output_path = 'merge-test-out.wav'
merge(input1_path, input2_path, output_path, duration = 7)
