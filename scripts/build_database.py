import os
import pickle
import numpy as np
import torch
import faiss
from feature_extraction import extract_features

# Detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using {device.upper()} for feature extraction and FAISS indexing.")

def build_feature_database(folder):
    """Extract and store deep audio features file by file, ensuring correct FAISS indexing."""
    feature_db = {}
    feature_list = []
    song_names = []
    
    for file in sorted(os.listdir(folder)):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            try:
                features = extract_features(file_path, model="pytorch", device=device)
                
                if features is None or not isinstance(features, np.ndarray):
                    print(f"⚠️ Skipping {file_path} - Extraction failed, returning None.")
                    continue
                
                if features.ndim == 1:
                    features = features.reshape(1, -1)  # Ensure correct shape
                
                if features.shape[1] == 0:
                    print(f"⚠️ Skipping {file_path} - Empty feature vector.")
                    continue
                
                feature_db[file_path] = features
                feature_list.append(features)
                song_names.append(file_path)
                print(f"✅ Processed: {file_path} - Feature Shape: {features.shape}")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
    
    if not feature_list:
        print("❌ No valid features extracted! Exiting.")
        return
    
    features = np.vstack(feature_list).astype(np.float32)  # Stack for FAISS
    d = features.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(features)
    
    with open("data/features.pkl", "wb") as f:
        pickle.dump((feature_db, index, song_names), f)
    print(f"✅ Feature database saved with {len(song_names)} songs!")

if __name__ == "__main__":
    build_feature_database("data/processed")
