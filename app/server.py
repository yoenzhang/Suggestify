from flask import Flask, request, jsonify
from match_audio_faiss import find_similar_songs

app = Flask(__name__)

@app.route("/match", methods=["POST"])
def match_song():
    """API endpoint to find similar songs based on uploaded audio."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files["audio"]
    file_path = "data/processed/temp.wav"
    file.save(file_path)

    matches = find_similar_songs(file_path, k=5)
    return jsonify({"suggestions": matches})

if __name__ == "__main__":
    app.run(debug=True)
