import os
import numpy as np
import torch
import faiss
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from feature_extraction import extract_features

FAISS_PATH = "data/faiss_index.pkl"

def process_file(file_path):
    try:
        features = extract_features(file_path, model="torchopenl3")

        if features is None or not isinstance(features, np.ndarray):
            print(f"⚠️ Skipping {file_path} - Extraction failed.")
            return None, None

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] == 0:
            print(f"⚠️ Skipping {file_path} - Empty feature vector.")
            return None, None

        # Normalize for FAISS indexing
        faiss.normalize_L2(features)
        return features.astype(np.float32), file_path

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def save_faiss_index(index, song_names):
    """Save FAISS index and processed song names."""
    faiss.write_index(index, FAISS_PATH)
    with open(FAISS_PATH.replace(".pkl", "_names.pkl"), "wb") as f:
        pickle.dump(song_names, f)
    print(f"✅ FAISS index saved: {FAISS_PATH}")

def load_faiss_index():
    """Load FAISS index and processed song names."""
    if os.path.exists(FAISS_PATH) and os.path.exists(FAISS_PATH.replace(".pkl", "_names.pkl")):
        index = faiss.read_index(FAISS_PATH)
        with open(FAISS_PATH.replace(".pkl", "_names.pkl"), "rb") as f:
            song_names = pickle.load(f)
        print(f"Resuming from FAISS index: {FAISS_PATH} ({len(song_names)} songs indexed)")
        return index, set(song_names)
    else:
        print(f"⚠️ No existing FAISS index found. Starting fresh.")
        return faiss.IndexFlatL2(512), set()

def build_feature_database(folder, batch_size=100, max_workers=3):
    """Extract and store deep audio features, ensuring correct FAISS indexing with batch updates."""
    index, processed_files = load_faiss_index()

    feature_list = []

    # Convert set to list for ordering
    song_names = list(processed_files)

    files = [os.path.join(folder, file) for file in sorted(os.listdir(folder)) if file.endswith(".wav")]
    files_to_process = [file for file in files if file not in processed_files]

    if not files_to_process:
        print("All files already indexed. Nothing to process.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file): file for file in files_to_process}

        for future in as_completed(futures):
            features, file_path = future.result()
            if features is not None and file_path is not None:
                feature_list.append(features)
                song_names.append(file_path)

                if len(feature_list) >= batch_size:
                    batch_features = np.vstack(feature_list)
                    index.add(batch_features)
                    save_faiss_index(index, song_names)
                    feature_list = []  # Reset batch list

    if len(feature_list) > 0:
        batch_features = np.vstack(feature_list)
        index.add(batch_features)
        save_faiss_index(index, song_names)

    print(f"Final FAISS index update! Total FAISS songs: {index.ntotal} (Expected: {len(song_names)})")

if __name__ == "__main__":
    build_feature_database("data/processed", batch_size=100, max_workers=3)
