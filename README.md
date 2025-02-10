# **Suggestify: AI-Powered Music Recommendation App**

## **Overview**

Suggestify is an AI-powered music recognition and recommendation system that:

- Records audio and converts it into deep audio embeddings.
- Compares it against a large music database using FAISS indexing.
- Returns a ranked list of similar songs based on audio similarity.

## **Project Structure**

```plaintext
suggestify/
│── data/                  # Audio dataset
│   ├── raw/               # Original MP3/WAV files (ignored in .gitignore)
│   ├── processed/         # Converted WAV files (16kHz, mono)
│   ├── features.pkl       # Saved audio embeddings
│   ├── faiss_index        # FAISS index for fast similarity search
│── models/                # Machine learning models (to be implemented)
│── scripts/               # Core functionality
│   ├── preprocess.py      # Convert audio to WAV
│   ├── feature_extraction.py # Extract deep embeddings using TorchOpenL3
│   ├── build_database.py  # Store features in FAISS
│   ├── match_audio.py     # Find similar songs
│── app/                   # Flask API backend (WIP)
│   ├── server.py          # API for handling requests
│── frontend/              # React Native mobile app (WIP)
│── notebooks/             # Jupyter notebooks for testing
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## **Quick Start**

1. **Install Dependencies**

```sh
pip install -r requirements.txt
```

2. **Convert MP3s to WAV**

```sh
python scripts/preprocess.py
```

3. **Extract Audio Features**

```sh
python scripts/feature_extraction.py
```

4. **Build Feature Database**

```sh
python scripts/build_database.py
```

5. **Match a New Audio File**

```sh
python scripts/match_audio.py
```

6. **Run API Server**

** Work in progress **

```sh
python app/server.py
```

## **Features**

- Fast audio matching using FAISS indexing.
- Deep audio embeddings with TorchOpenL3.
- REST API backend for mobile integration.
- Scalable for large music datasets.

## **Notes**

- The model is trained on approximately 8GB of music data containing 8000 30-second snippets.
- Data files in `data/raw/` are ignored in Git. However, I have uploaded my .pkl files for quick runs upon pulling.
- Ensure all audio files are processed before running similarity matching.
