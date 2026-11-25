# SignSpeak AI - Model Evaluation Script

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "dataset" / "raw" / "alphabet_landmarks.csv"
MODELS_DIR = PROJECT_ROOT / "models"

def load_data():
    """Load dataset from CSV file."""
    if not DATASET_PATH.exists():
        print(f"Dataset not found at: {DATASET_PATH}")
        return None, None
    
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Separate features and labels
    X = df.iloc[:, 4:].values  # Features start from column 5 (index 4)
    y = df.iloc[:, 1].values   # Labels are in column 2 (index 1)
    
    return X, y

def evaluate_model(X, y, model_path='models/signspeak_mlp_deduplicated_v3_controlled.pkl', scaler_path='models/scaler.pkl'):
    """Evaluate trained model on dataset."""
    # Load model and scaler
    try:
        model = joblib.load(MODELS_DIR / model_path)
        scaler = joblib.load(MODELS_DIR / scaler_path)
        print(f"Loaded model: {model_path}")
        print(f"Loaded scaler: {scaler_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

def main():
    print("=" * 60)
    print("SignSpeak AI - Model Evaluation")
    print("=" * 60)
    
    # Load data
    X, y = load_data()
    if X is None:
        return
    
    # Evaluate model
    results = evaluate_model(X, y)
    if results is None:
        return
    
    # Print results
    print(f"\nTest Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print("\nClassification Report:")
    target_names = [chr(65 + i) for i in range(26)]  # A-Z
    print(pd.DataFrame(results['classification_report']).transpose())
    
    # Save results
    output_path = PROJECT_ROOT / "outputs" / "evaluation_results.json"
    import json
    output_data = {
        'accuracy': float(results['accuracy']),
        'classification_report': results['classification_report'],
        'confusion_matrix': results['confusion_matrix']
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_path}")

if __name__ == "__main__":
    main()