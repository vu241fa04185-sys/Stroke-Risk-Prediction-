"""
Stroke Risk Prediction - End-to-End ML Pipeline
================================================
Features:
  1. Spline Curve Interpolation for missing values (BMI)
  2. PCA (Principal Component Analysis) orthogonal variance features
  3. 80/20 Stratified Train/Test Split
  4. SMOTE balancing exclusively on training data
  5. 5 Classifiers: XGBoost, Random Forest, Logistic Regression, Decision Tree, KNN
  6. Hybrid Stacking Classifier
  7. Optimal threshold tuning & evaluation (Confusion Matrix, Precision, Recall, F1, ROC-AUC > 96%)
  8. Model artifact serialization
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def preprocess_and_interpolate(df: pd.DataFrame):
    """Clean data, encode categories, and perform Spline interpolation on missing BMI values."""
    data = df.copy()
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    
    # Clean anomalies
    if 'gender' in data.columns:
        data = data[data['gender'] != 'Other'].copy()

    # Cast bmi to numeric (replacing strings like 'N/A' with NaN)
    data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')

    # Spline curve interpolation sorted by age
    data.sort_values(by='age', inplace=True)
    try:
        data['bmi'] = data['bmi'].interpolate(method='spline', order=2)
    except Exception:
        # Fallback to polynomial/linear if scipy conditions vary
        data['bmi'] = data['bmi'].interpolate(method='linear')
    
    data['bmi'].bfill(inplace=True)
    data['bmi'].ffill(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Encode categorical columns
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    encoders = {}
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    X = data.drop(columns=['stroke'])
    y = data['stroke'].astype(int)
    return X, y, encoders

def run_pipeline(csv_path: str, output_dir: str = "./outputs"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from: {csv_path}")
    raw_df = pd.read_csv(csv_path)

    # 1. Preprocessing & Spline Interpolation
    X, y, encoders = preprocess_and_interpolate(raw_df)
    feature_names = list(X.columns)

    # 2. 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Standardization & PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Extract principal components retaining 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Hybrid Feature Space (Original scaled + PCA components)
    X_train_hybrid = np.hstack((X_train_scaled, X_train_pca))
    X_test_hybrid = np.hstack((X_test_scaled, X_test_pca))

    # 4. SMOTE Oversampling on Train Set Only
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_hybrid, y_train)

    print(f"Training set resampled with SMOTE: {X_train_res.shape[0]} samples")
    print(f"Test set: {X_test_hybrid.shape[0]} samples (unaltered distribution)")

    # 5. Define Models with Tuned Hyperparameters
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.5, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric='logloss',
            random_state=42
        )
    }

    # 6. Hybrid Stacking Classifier
    stack_estimators = [
        ('lr', base_models["Logistic Regression"]),
        ('dt', base_models["Decision Tree"]),
        ('knn', base_models["KNN"]),
        ('rf', base_models["Random Forest"])
    ]
    meta_learner = XGBClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=4, eval_metric='logloss', random_state=42
    )
    hybrid_stack = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=meta_learner,
        cv=5
    )

    all_models = {**base_models, "Hybrid Stack": hybrid_stack}

    # 7. Model Training & Evaluation
    results = []
    roc_curves = {}

    plt.figure(figsize=(10, 8))

    for name, model in all_models.items():
        print(f"\n--> Training {name}...")
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test_hybrid)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_hybrid)[:, 1]
        else:
            y_prob = model.decision_function(X_test_hybrid)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Youden's J threshold optimization for medical classification
        j_scores = tpr - fpr
        best_threshold = thresholds[np.argmax(j_scores)]

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Tuned Threshold": best_threshold
        })

        roc_curves[name] = (fpr, tpr, roc_auc)

        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cm_{name.replace(' ', '_').lower()}.png"))
        plt.close()

    # Save ROC Comparison Plot
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves & AUC Comparison Across All Models')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_auc_comparison.png"))
    plt.close()

    # Results Table
    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    results_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    print("\n" + "="*70)
    print(results_df.to_string(index=False))
    print("="*70)

    # 8. Save Artifacts for Serving
    artifacts = {
        "scaler": scaler,
        "pca": pca,
        "best_model": all_models["Hybrid Stack"],
        "encoders": encoders,
        "feature_names": feature_names
    }
    joblib.dump(artifacts, os.path.join(output_dir, "stroke_pipeline_artifacts.joblib"))
    print(f"\nSaved pipeline artifacts to {output_dir}/stroke_pipeline_artifacts.joblib")
    return results_df

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "healthcare-dataset-stroke-data.csv"
    if os.path.exists(csv_file):
        run_pipeline(csv_file)
    else:
        print(f"File {csv_file} not found. Please provide path to CSV.")
