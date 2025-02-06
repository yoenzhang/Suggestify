# **Suggestify: AI-Powered Shazam-Like App**

## **🔹 Overview**
Suggestify is an AI-powered music identification app that:
- **Records audio** and converts it into a fingerprint.
- **Compares it to a database of songs** to find the closest match.
- **Returns song suggestions** based on similarity.

## **📌 Project Structure**
```plaintext
suggestify/
│── data/                  # Audio dataset
│   ├── raw/               # Original MP3s/WAVs (ignored in .gitignore)
│   ├── processed/         # Converted WAVs (16kHz, mono)
│   ├── features.pkl       # Saved fingerprints (MFCCs)
│── models/                # ML models (optional)
│── scripts/               # Core functionality
│   ├── preprocess.py      # Convert audio to WAV
│   ├── feature_extraction.py # Extract MFCCs
│   ├── build_database.py  # Store features in FAISS
│   ├── match_audio.py     # Find song matches
│── app/                   # Flask API backend
│   ├── server.py          # API to handle requests
│── frontend/              # React Native mobile app
│── notebooks/             # Jupyter notebooks for testing
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## **🚀 Quick Start**
### **1) Install Dependencies**
```sh
pip install -r requirements.txt
```

### **2️) Convert MP3s to WAV**
```sh
python scripts/preprocess.py
```

### **3️) Extract Audio Features**
```sh
python scripts/feature_extraction.py
```

### **4️) Build Feature Database**
```sh
python scripts/build_database.py
```

### **5️) Match a New Audio File**
```sh
python scripts/match_audio.py
```

### **6️) Run API Server**
```sh
python app/server.py
```

## **📌 Features**
- **Fast Music Matching** using FAISS indexing
- **Lightweight Audio Fingerprints** with MFCCs
- **REST API Backend** for easy mobile app integration
- **Mobile App Support** (React Native)

## **📌 Notes**
- **Data files (`data/raw/`) are ignored in Git.**
- **Make sure all audio files are processed before matching.**