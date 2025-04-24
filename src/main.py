import os
import argparse
from data.prepare_dataset import prepare_dataset
from data.data_preprocessing import preprocess_dataset
from data.feature_extraction import create_feature_dataframe
from training.train import load_data, train_and_evaluate_model
from utils.record_audio import record_voice
from utils.model_utils import load_model
from training.predict import classify_recording

def run_full_pipeline():
    print("Running full pipeline...")
    # Data preparation (includes cleaning)
    valid_files, empty_files = prepare_dataset()
    print(f"Number of valid files: {len(valid_files)}")
    print(f"Number of empty files: {len(empty_files)}")

    # Preprocessing
    preprocess_dataset()

    # Feature extraction
    features_df = create_feature_dataframe()
    print("Feature DataFrame shape:", features_df.shape)

    # Training and evaluation
    X_train, X_test, y_train, y_test, scaler = load_data()
    train_and_evaluate_model(X_train, y_train, X_test, y_test, scaler)

def record_and_classify():
    print("Recording and classifying...")
    recording_path = "data/external/recording.wav"
    record_voice(recording_path)
    
    scaler_path = "models/scaler.pkl"
    model_path = "models/logistic_regression_sklearn.pkl"
    if not (os.path.exists(scaler_path) and os.path.exists(model_path)):
        print("Scaler or model not found. Please run the full pipeline first.")
        return
    
    scaler = load_model(scaler_path)
    model = load_model(model_path)
    feature_columns = [f"mfcc_{i}" for i in range(13)] + ["spectral_centroid", "zcr", "rolloff"] + [f"chroma_{i}" for i in range(12)]
    predicted_class = classify_recording(recording_path, scaler, model, feature_columns)
    print("Predicted class (0/1):", predicted_class)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI project pipeline or record and classify voice.")
    parser.add_argument("--mode", choices=["full", "record"], default="full", help="Mode: 'full' for pipeline, 'record' for voice classification.")
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    if args.mode == "full":
        run_full_pipeline()
    elif args.mode == "record":
        record_and_classify()