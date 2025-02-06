import os
import pickle
from feature_extraction import extract_mfcc

feature_db = {}

def build_feature_database(folder):
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            feature_db[file] = extract_mfcc(file_path)

    # Save database
    with open("data/features.pkl", "wb") as f:
        pickle.dump(feature_db, f)

    print("Feature database saved!")

build_feature_database("data/processed")
