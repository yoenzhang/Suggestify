import os
import pickle
import numpy as np
import torch
import faiss

# Detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using {device.upper()} for feature extraction and FAISS indexing.")

def find_similar_songs(query_file, k=5):
    from feature_extraction import extract_features  # Import inside function to avoid circular import

    try:
        with open("data/features.pkl", "rb") as f:
            feature_db = pickle.load(f)  # Load the dictionary directly
    except FileNotFoundError:
        print("❌ Error: features.pkl file not found.")
        return []
    except Exception as e:
        print(f"❌ Error loading features.pkl: {e}")
        return []

    song_names = list(feature_db.keys())
    
    try:
        query_features = extract_features(query_file, model="torchopenl3", device=device).reshape(1, -1).astype(np.float32)
    except Exception as e:
        print(f"❌ Error extracting features from query file: {e}")
        return []

    # Convert feature dictionary to numpy array for FAISS indexing
    feature_matrix = np.vstack(list(feature_db.values())).astype(np.float32)
    
    # Create FAISS index
    d = feature_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(feature_matrix)
    
    _, idx = index.search(query_features, k)  # Run FAISS search
    return [song_names[i] for i in idx[0]]

if __name__ == "__main__":
    matches = find_similar_songs("data/test/4x4.wav", k=5)
    print(f"Top 5 similar songs: {matches}")