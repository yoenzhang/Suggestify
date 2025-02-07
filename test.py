import torch

print("🔹 PyTorch CUDA Available:", torch.cuda.is_available())
print("🔹 Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print(f"🔹 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU detected, falling back to CPU.")




# from pydub.utils import which
# print(which("ffmpeg"))

# import pickle

# with open("data/features.pkl", "rb") as f:
#     feature_db = pickle.load(f)

# print("Number of songs:", len(feature_db))
# print("Feature vector shape:", next(iter(feature_db.values())).shape if feature_db else "EMPTY")
