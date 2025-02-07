import os
from pydub import AudioSegment
from pydub.utils import which

# Manually Set FFmpeg Paths
AudioSegment.converter = "C:/FFmpeg/bin/ffmpeg.exe"
AudioSegment.ffmpeg = "C:/FFmpeg/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/FFmpeg/bin/ffprobe.exe"

# Get the absolute path of the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def convert_to_wav(input_folder, output_folder):
    input_folder = os.path.join(PROJECT_ROOT, input_folder)  # Ensure absolute path
    output_folder = os.path.join(PROJECT_ROOT, output_folder)  # Ensure absolute path

    if not os.path.exists(input_folder):
        print(f"❌ ERROR: Input folder does not exist: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(input_folder):  # Recursively walk through all subdirectories
        for file in files:
            if file.endswith((".mp3", ".wav")):
                input_path = os.path.join(root, file)

                # Check if file exists
                if not os.path.exists(input_path):
                    print(f"❌ ERROR: File not found - {input_path}")
                    continue
                
                print(f"Processing: {input_path}")

                output_filename = os.path.splitext(file)[0] + ".wav"
                output_path = os.path.join(output_folder, output_filename)

                try:
                    audio = AudioSegment.from_file(input_path, format="mp3" if file.endswith(".mp3") else "wav")

                    # Convert to Mono, 44.1kHz (required for OpenL3), 16-bit PCM WAV
                    audio = audio.set_channels(1).set_frame_rate(44100)
                    audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])

                except Exception as e:
                    print(f"❌ Error processing {input_path}: {e}")

# Run preprocessing for the entire dataset
convert_to_wav("data/raw/fma_small", "data/processed")
