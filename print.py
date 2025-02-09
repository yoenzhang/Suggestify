import pickle
import numpy as np
import faiss

# Paths to saved files
features_path = "data/features.pkl"
faiss_path = "data/faiss_index.pkl"

def verify_data():
    """Verify that features.pkl and faiss_index.pkl contain matching data."""
    
    # ✅ Load feature database
    try:
        with open(features_path, "rb") as f:
            feature_db, song_names = pickle.load(f)
        print(f"✅ Loaded {len(song_names)} songs from {features_path}")
    except Exception as e:
        print(f"❌ Error loading {features_path}: {e}")
        return

    # ✅ Load FAISS index
    try:
        index = faiss.read_index(faiss_path)
        print(f"✅ FAISS Index contains {index.ntotal} songs.")
    except Exception as e:
        print(f"❌ Error loading {faiss_path}: {e}")
        return

    if len(song_names) != index.ntotal:
        print(f"❌ WARNING: FAISS index count ({index.ntotal}) does not match feature database count ({len(song_names)})!")

    # ✅ Verify first 5 songs
    print("\n🔍 **Verifying first 5 songs:**\n")
    for i in range(min(5, len(song_names))):
        song = song_names[i]
        feature_vector = feature_db[song].flatten()  # Ensure it is (512,)

        # Retrieve FAISS stored vector with the correct shape
        faiss_vector = np.zeros((index.d,), dtype=np.float32)  # Ensure shape (512,)
        index.reconstruct(i, faiss_vector)

        print(f"🎵 {i+1}. {song}")
        print(f"   🔹 Feature Shape (features.pkl): {feature_vector.shape}")
        print(f"   🔹 Feature Shape (FAISS): {faiss_vector.shape}")

        # Compare vectors using cosine similarity
        cosine_sim = np.dot(feature_vector, faiss_vector) / (
            np.linalg.norm(feature_vector) * np.linalg.norm(faiss_vector)
        )

        print(f"   📊 Cosine Similarity: {cosine_sim:.4f}\n")

    print("✅ Verification complete!")

# Run verification
verify_data()
