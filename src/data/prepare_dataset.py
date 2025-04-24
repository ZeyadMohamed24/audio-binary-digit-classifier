import os
import shutil
import librosa
import numpy as np
from config import RAW_PATH, INTERIM_PATH, REJECTED_PATH


def list_wav_files(path):
    files = os.listdir(path)
    return [f for f in files if f.endswith(".wav")]


def validate_file(file_path, energy_threshold=0.001, min_duration=0.1):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        energy = np.sum(audio**2) / len(audio)
        return energy >= energy_threshold and duration >= min_duration
    except Exception as e:
        print(f"Error in file {file_path}: {e}")
        return False


def prepare_dataset():
    os.makedirs(INTERIM_PATH, exist_ok=True)
    os.makedirs(REJECTED_PATH, exist_ok=True)
    wav_files = list_wav_files(RAW_PATH)
    for file_name in wav_files:
        shutil.copyfile(
            os.path.join(RAW_PATH, file_name), os.path.join(INTERIM_PATH, file_name)
        )

    valid_files = []
    empty_files = []
    for file_name in os.listdir(INTERIM_PATH):
        if file_name.endswith(".wav"):
            file_path = os.path.join(INTERIM_PATH, file_name)
            if validate_file(file_path):
                valid_files.append(file_name)
            else:
                empty_files.append(file_name)
                shutil.move(file_path, os.path.join(REJECTED_PATH, file_name))
    return valid_files, empty_files
