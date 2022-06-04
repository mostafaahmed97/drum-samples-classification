import sys
import pickle
import librosa
import helpers as helpers

with open("model.pickle", "rb") as handle:
    model = pickle.load(handle)

with open("scaler.pickle", "rb") as handle:
    scaler = pickle.load(handle)

args = sys.argv[1:]
if not len(args):
    print("Path for test file not provided")
    sys.exit()

signal, sr = librosa.load(args[0], mono=True, sr=44100)
processed_sig = helpers.preprocess_signal(signal, 200)
features = helpers.extract_features(processed_sig, sr)
features = [features]


pred = model.predict(scaler.transform(features))
print(f'The sample is a {pred[0]}')
