import pickle
import numpy as np
import faiss
import re

# Path to FAISS index and song names
faiss_path = "data/faiss_index.pkl"
song_names_path = "data/faiss_index_names.pkl"

def extract_numeric(filename):
    """Extract numeric value from filename for proper sorting."""
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else float('inf')

def list_last_50_faiss_songs():
    """Print total FAISS count and the last 50 songs stored in faiss_index.pkl in sorted order alongside their FAISS vectors."""

    # Load FAISS index
    try:
        index = faiss.read_index(faiss_path)
        total_songs = index.ntotal
        print(f"FAISS Index contains {total_songs} songs.")
    except Exception as e:
        print(f"Error loading {faiss_path}: {e}")
        return

    # Load song names
    try:
        with open(song_names_path, "rb") as f:
            song_names = pickle.load(f)
        print(f"Loaded {len(song_names)} song names.")
    except Exception as e:
        print(f"Error loading {song_names_path}: {e}")
        return

    if len(song_names) != total_songs:
        print(f"WARNING: FAISS index count ({total_songs}) does not match stored song count ({len(song_names)})!")

    # Sort song names numerically
    sorted_song_names = sorted(song_names, key=extract_numeric)

    # Extract last 50 songs
    last_50_songs = sorted_song_names[-50:]

    # Print last 50 songs and their FAISS vectors
    print("\n🔍 **Last 50 Sorted Songs and Their FAISS Vectors:**\n")
    for i, song in enumerate(last_50_songs, start=total_songs - 50 + 1):
        faiss_vector = np.zeros((index.d,), dtype=np.float32)
        index.reconstruct(i - 1, faiss_vector)  # Retrieve FAISS stored vector

        print(f"{i}. {song}")
        print(f"   🔹 FAISS Vector (first 5 values): {faiss_vector[:5]}\n")

# Run FAISS song listing
list_last_50_faiss_songs()
