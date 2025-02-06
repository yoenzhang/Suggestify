import pickle
import numpy as np
import faiss
from feature_extraction import extract_mfcc

# Load stored song features
with open("data/features.pkl", "rb") as f:
    feature_db = pickle.load(f)

song_names = list(feature_db.keys())
features = np.array(list(feature_db.values()), dtype=np.float32)

# Create FAISS index

# Number of MFCC coefficients
dimension = features.shape[1] 
index = faiss.IndexFlatL2(dimension)
index.add(features)

def find_closest_match(query_file):
    query_features = extract_mfcc(query_file).reshape(1, -1).astype(np.float32)
    _, idx = index.search(query_features, 1)
    return song_names[idx[0][0]]

# Test with an unknown file
if __name__ == "__main__":
    match = find_closest_match("data/processed/test_recording.wav")
    print(f"Closest match: {match}")
