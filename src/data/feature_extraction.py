import os
import pandas as pd
import librosa
import numpy as np
from config import PROCESSED_PATH, FEATURES_CSV_PATH


def extract_features(file_path, n_fft=512):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        n_fft = min(n_fft, len(audio))
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft)
        mfccs_mean = np.mean(mfccs, axis=1)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=n_fft
        )
        spectral_centroid_mean = np.mean(spectral_centroid)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft)
        rolloff_mean = np.mean(rolloff)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft)
        chroma_mean = np.mean(chroma, axis=1)
        return np.concatenate(
            [
                mfccs_mean,
                [spectral_centroid_mean],
                [zcr_mean],
                [rolloff_mean],
                chroma_mean,
            ]
        )
    except Exception as e:
        print(f"Error in file {file_path}: {e}")
        return None


def create_feature_dataframe():
    feature_columns = (
        [f"mfcc_{i}" for i in range(13)]
        + ["spectral_centroid", "zcr", "rolloff"]
        + [f"chroma_{i}" for i in range(12)]
    )
    features_list = []
    labels = []
    for file_name in os.listdir(PROCESSED_PATH):
        if file_name.endswith(".wav"):
            file_path = os.path.join(PROCESSED_PATH, file_name)
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
                labels.append(0 if file_name.startswith("0_") else 1)
    features_df = pd.DataFrame(features_list, columns=feature_columns)
    features_df["label"] = labels
    features_df.to_csv(FEATURES_CSV_PATH, index=False)
    return features_df
