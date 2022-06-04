import librosa
import numpy as np
import matplotlib.pyplot as plt

def plot_spectrogram(signal, sr):
    stft = librosa.stft(signal)
    stft_magnitude, _ = librosa.magphase(stft)

    mel_spectro = librosa.feature.melspectrogram(S=stft_magnitude, sr=sr)
    db_mel_spectro = librosa.amplitude_to_db(mel_spectro, ref=np.min)
    
    librosa.display.specshow(db_mel_spectro, x_axis='time', y_axis='mel')
    plt.colorbar(format="%+2.0f dB")
    plt.show()

def preprocess_signal(signal, max_ms):
    max_len = (44100 // 1000) * max_ms
    signal_len = len(signal)

    if signal_len > max_len:
        signal = signal[:max_len]
    else:
        pad_len = max_len - signal_len
        zeros = np.zeros(pad_len)
        signal = [*signal, *zeros]

    return np.array(signal)

def extract_features(signal, sr):
    zero_crossings = sum(librosa.zero_crossings(signal, pad=False))

    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_rollof = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]

    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=12)
    mfccs_mean = np.mean(mfccs, axis=1)

    features = np.hstack(
        (zero_crossings, spectral_centroid, spectral_rollof, mfccs_mean)
    )
    return np.array(features)