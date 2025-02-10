import os
import numpy as np
import torch
import faiss
import pickle
from feature_extraction import extract_features

FAISS_INDEX_PATH = "data/faiss_index.pkl"
SONG_NAMES_PATH = "data/faiss_index_names.pkl"

def load_faiss_index():
    """Load FAISS index and song names from file."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(SONG_NAMES_PATH):
        print("Error: FAISS index or song names file not found.")
        return None, None
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    # Extract stored song names from the pickle file
    with open(SONG_NAMES_PATH, "rb") as f:
        song_names = pickle.load(f)
    
    # Check if the index is valid
    if index.ntotal == 0 or not index.is_trained:
        print("Error: FAISS index is empty or untrained.")
        return None, None

    print(f"Loaded FAISS index with {index.ntotal} songs.")

    return index, song_names

def find_similar_songs(query_file, k=5): 
    # Load FAISS index and song names
    index, song_names = load_faiss_index()
    if index is None or song_names is None:
        return []

    try:
        query_features = extract_features(query_file, model="torchopenl3").astype(np.float32)
        query_features = query_features.reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features from query file: {e}")
        return []

    # Normalize query feature for FAISS search
    faiss.normalize_L2(query_features)

    # Run FAISS search
    distances, idx = index.search(query_features, k)

    # Retrieve song names and similarity scores
    results = [(song_names[int(i)], distances[0][j]) for j, i in enumerate(idx[0])]
    
    return results

if __name__ == "__main__":
    # Set environment variable to avoid OpenMP runtime error
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    matches = find_similar_songs("data/test/drink dont need no mix.wav", k=5)
    print("🎵 Top 5 similar songs:")
    for song, score in matches:
        print(f"  {song} (Similarity Score: {score:.4f})")