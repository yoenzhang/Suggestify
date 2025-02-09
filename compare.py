import faiss
import numpy as np
import pickle

# Paths to FAISS index and features.pkl
faiss_path = "data/faiss_index.pkl"
features_path = "data/features.pkl"

def compare_faiss_vectors():
    """Compare all FAISS indexed vectors and output file names for verification."""
    
    # ✅ Load FAISS index
    index = faiss.read_index(faiss_path)
    num_songs = index.ntotal

    print(f"✅ FAISS Index contains {num_songs} songs.")
    
    if num_songs < 2:
        print("❌ Not enough songs to compare.")
        return

    # ✅ Load song names from features.pkl
    try:
        with open(features_path, "rb") as f:
            feature_db, song_names = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading {features_path}: {e}")
        return
    
    if len(song_names) != num_songs:
        print(f"❌ WARNING: FAISS index count ({num_songs}) does not match feature database count ({len(song_names)})!")

    # ✅ Extract all FAISS stored vectors
    stored_vectors = np.zeros((num_songs, index.d), dtype=np.float32)
    for i in range(num_songs):
        index.reconstruct(i, stored_vectors[i])

    # ✅ Compute pairwise cosine similarity
    norm_vectors = stored_vectors / np.linalg.norm(stored_vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_vectors, norm_vectors.T)

    # ✅ Check for near-identical embeddings and output filenames
    print("\n🔍 **Pairwise Cosine Similarities:**\n")
    for i in range(num_songs):
        for j in range(i + 1, num_songs):  # Compare each pair once
            similarity = similarity_matrix[i, j]
            if similarity > 0.98:  # Flag highly similar embeddings
                print(f"⚠️ Songs {i} ({song_names[i]}) & {j} ({song_names[j]}) are very similar! Cosine Sim: {similarity:.4f}")

    print("✅ FAISS vector comparison complete!")

# Run the comparison
compare_faiss_vectors()
