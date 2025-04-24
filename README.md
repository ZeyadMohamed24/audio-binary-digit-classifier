# 🎙️ Audio Binary Digit Classifier

A **binary audio classification** system that distinguishes between spoken digits **"0"** and **"1"** using machine learning. This project handles raw `.wav` audio, extracts features, trains models (Naive Bayes, Logistic Regression), and supports **real-time audio recording** for prediction.

---

## 📁 Project Structure

```
audio-binary-digit-classifier/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── main.py               # Entry point for full pipeline or real-time recording
│   ├── config.py             # Data paths
│   ├── data/                 # Preprocessing and feature extraction scripts
│   ├── models/               # Custom and sklearn model implementations
│   ├── training/             # Model training and saving
│   ├── evaluation/           # Evaluation metrics and visualizations
│   └── utils/                # Audio processing utilities
├── data/
│   ├── raw/                  # Original audio files
│   ├── interim/              # Cleaned and validated audio
│   ├── processed/            # Extracted feature data
│   ├── rejected/             # Invalid or corrupt audio files
│   └── external/             # Recorded audio
├── reports/
│   ├── metrics.csv           # Model performance metrics
│   └── *.png                 # Confusion matrices, ROC curves, etc.
└── models/                   # Models saved as pkl 

```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <https://github.com/ZeyadMohamed24/audio-binary-digit-classifier.git>
cd audio-binary-digit-classifier
```

### 2. Create a Virtual Environment If needed

```bash
python -m venv venv
source venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation

- Place `.wav` files in `data/raw/`
- File names must begin with `0_` or `1_` to indicate their class

---

## 🚀 Usage

### Run the Full Pipeline

```bash
python src/main.py --mode full
```

This runs data cleaning, feature extraction, model training, and evaluation.

### Real-Time Recording & Prediction

```bash
python src/main.py --mode record
```

This records audio from your microphone and predicts the spoken digit.

---

## 🎯 Key Features

### 🔧 Preprocessing

- Silence and noise removal
- Normalization

### 🎵 Feature Extraction

- MFCCs
- Spectral Centroid
- Zero-Crossing Rate
- Spectral Roll-Off
- Chroma Features

### 🤖 Model Training

- Custom **Naive Bayes**
- Custom **Logistic Regression**
- Scikit-learn **Logistic Regression**
- **Bagging** ensemble method

### 📈 Evaluation

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrices
- ROC Curves

### 🎙️ Real-Time Prediction

- Record spoken digits
- Instantly classify between "0" and "1"

---

## 📊 Reports

- `reports/metrics.csv`: Evaluation metrics
- `reports/`: Visuals like confusion matrices and ROC curves

---
## 👥 Authors

- **Zeyad Mohamed** – [@ZeyadMohamed24](https://github.com/ZeyadMohamed24)
- **Aya Essam** – [@AyaEssam2004](https://github.com/AyaEssam2004)
- **Salsabeel Mohamed** – [@Salsabeel114](https://github.com/Salsabeel114)


## 🙏 Acknowledgments

Thanks to the creators of the **Free Spoken Digit Dataset (FSDD)** for making this project possible.
