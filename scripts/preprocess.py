import os
from pydub import AudioSegment

def convert_to_wav(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".mp3") or file.endswith(".wav"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".mp3", ".wav"))

            audio = AudioSegment.from_file(input_path)
             # Convert to Mono, 16kHz, standardized format across all songs
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(output_path, format="wav")

            print(f"Converted: {file} → {output_path}")

convert_to_wav("data/raw", "data/processed")
