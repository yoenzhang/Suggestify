import torchopenl3
import torch
import soundfile as sf
import librosa
import numpy as np
import os
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load TorchOpenL3 Model
MODEL = torchopenl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=512)

def extract_torchopenl3_features(file_path):
    """Extract deep audio features using TorchOpenL3."""
    try:
        # Load audio
        audio, sr = sf.read(file_path)
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000, res_type='kaiser_best')
        
        # Ensure valid audio length
        if len(audio) < 1000:  # Arbitrary threshold to avoid too-short clips
            print(f"⚠️ Skipping {file_path} - Too short for meaningful features.")
            return np.zeros((1, 512), dtype=np.float32)  # Dummy feature vector
        
        # Extract TorchOpenL3 features
        embedding, timestamps = torchopenl3.get_audio_embedding(audio, sr, model=MODEL, hop_size=1)
        
        # Debugging: Print raw shape
        print(f"🔍 Raw extracted feature shape before reshaping: {embedding.shape}")
        
        # Ensure correct FAISS shape
        features = torch.mean(embedding, dim=1).cpu().numpy().reshape(1, -1).reshape(1, -1)  # Take mean to get a fixed (1, 512) shape
        return features
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return np.zeros((1, 512), dtype=np.float32)  # Return dummy vector on failure

def extract_features(file_path, model="torchopenl3"):
    """Wrapper for feature extraction."""
    if model == "torchopenl3":
        return extract_torchopenl3_features(file_path)
    else:
        raise ValueError("Invalid model type. Choose 'torchopenl3'")

def save_features(features_dict, output_file="data/features.pkl"):
    """Save extracted features to a file."""
    with open(output_file, "wb") as f:
        pickle.dump(features_dict, f)
    print(f"✅ Features saved to {output_file}")

if __name__ == "__main__":
    processed_folder = "data/processed"
    features_dict = {}
    
    if not os.path.exists(processed_folder):
        print(f"❌ ERROR: Processed folder not found: {processed_folder}")
    else:
        for file in sorted(os.listdir(processed_folder)):
            if file.endswith(".wav"):
                file_path = os.path.join(processed_folder, file)
                features = extract_features(file_path, model="torchopenl3")
                features_dict[file] = features
                print(f"✅ Extracted Features Shape for {file}: {features.shape}")
        
        # Save all extracted features
            save_features(features_dict)
