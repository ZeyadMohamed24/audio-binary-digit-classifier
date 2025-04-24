from data.data_preprocessing import preprocess_audio
from data.feature_extraction import extract_features
import pandas as pd
import os


def classify_recording(recording_path, scaler, model, feature_columns):
    processed_path = "data/external/temp_recording.wav"
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    preprocess_audio(recording_path, processed_path)
    features = extract_features(processed_path)
    if features is None:
        print("Error extracting features.")
        return None
    input_df = pd.DataFrame([features], columns=feature_columns)
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)
