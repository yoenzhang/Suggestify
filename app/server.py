from flask import Flask, request, jsonify
from scripts.match_audio import find_closest_match

app = Flask(__name__)

@app.route("/match", methods=["POST"])
def match_song():
    file = request.files["audio"]
    file_path = "data/processed/temp.wav"
    file.save(file_path)

    match = find_closest_match(file_path)
    return jsonify({"match": match})

if __name__ == "__main__":
    app.run(debug=True)
