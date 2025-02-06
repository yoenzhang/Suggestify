import librosa
import numpy as np

# Convert to a single feature vector
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)  

# Test Extraction
if __name__ == "__main__":
    test_file = "data/processed/sample.wav"
    features = extract_mfcc(test_file)
    print("MFCC Features:", features)
