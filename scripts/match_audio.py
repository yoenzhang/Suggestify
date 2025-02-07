import os
import pickle
import numpy as np
import torch
import faiss
from feature_extraction import extract_features

# Detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using {device.upper()} for feature extraction and FAISS indexing.")

def find_similar_songs(query_file, k=5):
    with open("data/features.pkl", "rb") as f:
        feature_db, index = pickle.load(f)
    
    song_names = list(feature_db.keys())
    query_features = extract_features(query_file, model="pytorch", device=device).reshape(1, -1).astype(np.float32)
    
    _, idx = index.search(query_features, k)  # Run FAISS search on GPU
    return [song_names[i] for i in idx[0]]

if __name__ == "__main__":
    matches = find_similar_songs("data/test/4x4.wav", k=5)
    print(f"Top 5 similar songs: {matches}")
