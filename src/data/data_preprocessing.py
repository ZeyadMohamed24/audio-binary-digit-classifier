import os
import librosa
import soundfile as sf
import numpy as np
from config import INTERIM_PATH, PROCESSED_PATH


def preprocess_audio(file_path, output_path):
    audio, sr = librosa.load(file_path, sr=None)
    # Noise reduction
    noise_clip = audio[: int(sr * 0.01)]
    noise_mean = np.mean(np.abs(noise_clip))
    audio_clean = np.where(np.abs(audio) > noise_mean, audio, 0)
    # Silence removal
    audio_trimmed, _ = librosa.effects.trim(audio_clean, top_db=10)
    # Normalization
    audio_normalized = audio_trimmed / np.max(np.abs(audio_trimmed))
    sf.write(output_path, audio_normalized, sr)


def preprocess_dataset():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    for file_name in os.listdir(INTERIM_PATH):
        if file_name.endswith(".wav"):
            input_file = os.path.join(INTERIM_PATH, file_name)
            output_file = os.path.join(PROCESSED_PATH, file_name)
            preprocess_audio(input_file, output_file)
