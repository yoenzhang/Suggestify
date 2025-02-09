# import torch

# print("🔹 PyTorch CUDA Available:", torch.cuda.is_available())
# print("🔹 Number of GPUs:", torch.cuda.device_count())

# if torch.cuda.is_available():
#     print(f"🔹 Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("⚠️ No GPU detected, falling back to CPU.")


# import pickle
# import numpy as np

# # Load extracted features
# with open("data/features.pkl", "rb") as f:
#     features_dict = pickle.load(f)

# # Select two different songs
# song1 = "000002.wav"
# song2 = "000005.wav"

# # Get their feature vectors
# vec1 = features_dict[song1]
# vec2 = features_dict[song2]

# # Compute cosine similarity
# cosine_sim = np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# print(f"Cosine Similarity between {song1} and {song2}: {cosine_sim}")


# from pydub.utils import which
# print(which("ffmpeg"))

# import pickle

# with open("data/features.pkl", "rb") as f:
#     feature_db = pickle.load(f)

# print("Number of songs:", len(feature_db))
# print("Feature vector shape:", next(iter(feature_db.values())).shape if feature_db else "EMPTY")
