import os
import numpy as np
import torch
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from feature_extraction import extract_features

def process_file(file_path):
    try:
        features = extract_features(file_path, model="torchopenl3")

        if features is None or not isinstance(features, np.ndarray):
            print(f"⚠️ Skipping {file_path} - Extraction failed.")
            return None, None

        if features.ndim == 1:
            features = features.reshape(1, -1)  # Ensure correct shape

        if features.shape[1] == 0:
            print(f"⚠️ Skipping {file_path} - Empty feature vector.")
            return None, None

        # ✅ Log low variance but do not skip
        mean_value = np.mean(features)
        variance = np.var(features)
        if variance < 1e-6:
            print(f"⚠️ Warning: {file_path} has low variance ({variance:.6f}) but will still be indexed.")

        faiss.normalize_L2(features)  # Normalize for efficient indexing
        return features.astype(np.float32), file_path

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None, None

def build_feature_database(folder, batch_size=500, max_workers=2):
    """Extract and store deep audio features, ensuring correct FAISS indexing with batch updates."""
    faiss_path = "data/faiss_index.pkl"

    # ✅ Always start fresh
    print(f"⚠️ Overwriting existing {faiss_path} to prevent duplicates.")
    if os.path.exists(faiss_path):
        os.remove(faiss_path)

    # ✅ Create new FAISS index
    index = faiss.IndexFlatL2(512)
    faiss.write_index(index, faiss_path)
    print(f"✅ Created new FAISS index: {faiss_path}")

    feature_list = []  # Stores features in memory before adding to FAISS
    song_names = []  # Track processed filenames

    files = [os.path.join(folder, file) for file in sorted(os.listdir(folder)) if file.endswith(".wav")]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file) for file in files]

        for future in as_completed(futures):
            features, file_path = future.result()
            if features is not None and file_path is not None:
                feature_list.append(features)
                song_names.append(file_path)

                # ✅ Batch process FAISS every `batch_size` files
                if len(feature_list) >= batch_size:
                    batch_features = np.vstack(feature_list)
                    index.add(batch_features)
                    faiss.write_index(index, faiss_path)
                    print(f"✅ FAISS index updated! Total: {index.ntotal} songs.")
                    feature_list = []  # Reset batch list

    # ✅ Final Save to FAISS
    if len(feature_list) > 0:
        batch_features = np.vstack(feature_list)
        index.add(batch_features)
        faiss.write_index(index, faiss_path)
        print(f"✅ Final FAISS index update! Total FAISS songs: {index.ntotal} (Expected: {len(song_names)})")

    print("✅ Database build complete!")

if __name__ == "__main__":
    build_feature_database("data/processed", batch_size=500, max_workers=2)