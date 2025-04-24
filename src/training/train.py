from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.stats import mode
from config import FEATURES_CSV_PATH
from evaluation.evaluate import evaluate_model
from models.naive_bayes import Gaussian_Naive_Bayes
from models.logistic_regression import LogisticRegressionGD
from sklearn.linear_model import LogisticRegression
from utils.model_utils import save_model

def load_data():
    df = pd.read_csv(FEATURES_CSV_PATH)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=48
    )
    save_model(scaler, "models/scaler.pkl")
    return X_train, X_test, y_train, y_test, scaler

def train_model(model_class, X_train, y_train):
    model = model_class()
    model.fit(X_train, y_train)
    return model

def bootstrap_sample(X, y, n_samples):
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    return X[indices], y[indices]

def predict_bagging(models, X):
    predictions = np.array([model.predict(X) for model in models])
    final_predictions, _ = mode(predictions, axis=0)
    return final_predictions.flatten()

def apply_bagging(X_train, y_train, X_test, y_test, n_estimators=10, model_class=None, model_name=""):
    bootstrap_datasets = [
        bootstrap_sample(X_train, y_train, len(X_train)) for _ in range(n_estimators)
    ]
    models = []
    for X_sample, y_sample in bootstrap_datasets:
        model = train_model(model_class, X_sample, y_sample)
        models.append(model)
    predictions = predict_bagging(models, X_test)
    return evaluate_model(y_test, predictions, name=f"{model_name} Bagging")

def train_and_evaluate_model(X_train, y_train, X_test, y_test, scaler):
    metrics_list = []
    
    # Naive Bayes
    clf_nb = train_model(Gaussian_Naive_Bayes, X_train, y_train)
    save_model(clf_nb, "models/naive_bayes.pkl")
    h_nb = clf_nb.predict(X_test)
    metrics_nb = evaluate_model(y_test, h_nb, name="Gaussian Naive Bayes")
    metrics_list.append(metrics_nb)
    
    # Bagging with Naive Bayes
    metrics_nb_bagging = apply_bagging(
        X_train, y_train, X_test, y_test, n_estimators=10, 
        model_class=Gaussian_Naive_Bayes, model_name="Gaussian Naive Bayes"
    )
    metrics_list.append(metrics_nb_bagging)
    
    # Logistic Regression (custom)
    clf_lr_scratch = train_model(LogisticRegressionGD, X_train, y_train)
    save_model(clf_lr_scratch, "models/logistic_regression_custom.pkl")
    h_lr_scratch = clf_lr_scratch.predict(X_test)
    metrics_lr_scratch = evaluate_model(y_test, h_lr_scratch, name="Logistic Regression from Scratch")
    metrics_list.append(metrics_lr_scratch)
    
    # Bagging with Logistic Regression (custom)
    metrics_lr_scratch_bagging = apply_bagging(
        X_train, y_train, X_test, y_test, n_estimators=10, 
        model_class=LogisticRegressionGD, model_name="Logistic Regression from Scratch"
    )
    metrics_list.append(metrics_lr_scratch_bagging)
    
    # Logistic Regression (sklearn)
    clf_lr_sk = LogisticRegression()
    clf_lr_sk.fit(X_train, y_train)
    save_model(clf_lr_sk, "models/logistic_regression_sklearn.pkl")
    h_lr_sk = clf_lr_sk.predict(X_test)
    probs_lr_sk = clf_lr_sk.predict_proba(X_test)[:, 1]
    metrics_lr_sk = evaluate_model(y_test, h_lr_sk, y_pred_proba=probs_lr_sk, name="Logistic Regression (sklearn)")
    metrics_list.append(metrics_lr_sk)
    
    # Bagging with Logistic Regression (sklearn)
    metrics_lr_sk_bagging = apply_bagging(
        X_train, y_train, X_test, y_test, n_estimators=10, 
        model_class=LogisticRegression, model_name="Logistic Regression (sklearn)"
    )
    metrics_list.append(metrics_lr_sk_bagging)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("reports/metrics.csv", index=False)