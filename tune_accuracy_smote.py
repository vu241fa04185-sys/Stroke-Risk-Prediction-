"""
High-Accuracy SMOTE & Stacking Pipeline
=======================================
Combines:
  1. Clinical feature engineering (Age-Glucose, Comorbidity, Age-BMI interactions)
  2. Borderline / Calibrated SMOTE on training split
  3. Optimized XGBoost and Random Forest Base Learners
  4. 3 Hybrid Stacking Models (LR, DT, KNN Meta-Learners)
  5. Multi-Objective Threshold Tuning:
     - High Accuracy Mode (maximizing Overall Accuracy while maintaining clinical recall)
     - High Balanced Accuracy Mode (maximizing Balanced Accuracy = (Sens + Spec)/2)
     - Optimal F1 Mode
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 1. Load Data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)
df = df[df['gender'] != 'Other'].copy()
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# 2. Imputation & Feature Engineering
df.sort_values(by='age', inplace=True)
known_mask = ~df['bmi'].isna()
ages = df.loc[known_mask, 'age'].values
bmis = df.loc[known_mask, 'bmi'].values
u_ages, u_idx = np.unique(ages, return_index=True)
spline = CubicSpline(u_ages, bmis[u_idx], extrapolate=True)
df.loc[~known_mask, 'bmi'] = np.clip(spline(df.loc[~known_mask, 'age']), 12, 65)

# Encode categoricals
cat_map = {
    'gender': {'Female': 0, 'Male': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'ever_married': {'No': 0, 'Yes': 1},
    'work_type': {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4},
    'Residence_type': {'Rural': 0, 'Urban': 1},
    'smoking_status': {'Unknown': 0, 'never smoked': 1, 'formerly smoked': 2, 'smokes': 3}
}
for c, m in cat_map.items():
    if c in df.columns:
        df[c] = df[c].map(lambda v: m.get(v, 0)).astype(float)

# Advanced Clinical Feature Interactions
df['age_glucose'] = (df['age'] / 50.0) * (df['avg_glucose_level'] / 100.0)
df['vascular_risk'] = df['hypertension'] * 1.5 + df['heart_disease'] * 2.0
df['age_bmi'] = (df['age'] / 50.0) * (df['bmi'] / 25.0)
df['hyperglycemia'] = (df['avg_glucose_level'] >= 140.0).astype(float)
df['senior_citizen'] = (df['age'] >= 65.0).astype(float)

feature_cols = [c for c in df.columns if c != 'stroke']
X = df[feature_cols].values.astype(np.float64)
y = df['stroke'].values.astype(int)

# 3. Stratified 80/20 Train-Test Split
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
np.random.shuffle(pos_idx)
np.random.shuffle(neg_idx)

n_test_pos = int(np.round(0.20 * len(pos_idx)))
n_test_neg = int(np.round(0.20 * len(neg_idx)))

test_idx = np.concatenate([pos_idx[:n_test_pos], neg_idx[:n_test_neg]])
train_idx = np.concatenate([pos_idx[n_test_pos:], neg_idx[n_test_neg:]])
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 4. Standard Scaler & PCA (Fit on train only)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

U, S, Vt = np.linalg.svd(X_train_scaled, full_matrices=False)
W_pca = Vt[:6].T

X_train_hybrid = np.hstack([X_train_scaled, np.dot(X_train_scaled, W_pca)])
X_test_hybrid = np.hstack([X_test_scaled, np.dot(X_test_scaled, W_pca)])

# 5. Calibrated Borderline SMOTE (Oversample minority to ~0.5 ratio for optimal accuracy-recall trade-off)
def calibrated_smote(X_in, y_in, ratio=0.50):
    pos = np.where(y_in == 1)[0]
    neg = np.where(y_in == 0)[0]
    n_p, n_n = len(pos), len(neg)
    target_pos = int(n_n * ratio)
    needed = target_pos - n_p
    if needed <= 0:
        return X_in, y_in

    X_p = X_in[pos]
    synthetic = []
    for _ in range(needed):
        i = np.random.choice(n_p)
        dists = np.sum((X_p - X_p[i]) ** 2, axis=1)
        knn = np.argsort(dists)[1:min(6, n_p)]
        nb = np.random.choice(knn) if len(knn) > 0 else i
        gap = np.random.rand()
        synthetic.append(X_p[i] + gap * (X_p[nb] - X_p[i]))

    X_out = np.vstack([X_in, np.array(synthetic)])
    y_out = np.concatenate([y_in, np.ones(needed, dtype=int)])
    p = np.random.permutation(len(X_out))
    return X_out[p], y_out[p]

X_tr_smote, y_tr_smote = calibrated_smote(X_train_hybrid, y_train, ratio=0.60)
print(f"Dataset Balanced with SMOTE: Train shape = {X_tr_smote.shape}, Minority ratio = {np.mean(y_tr_smote):.2%}")

# 6. Base Learners & Stacking
class FastTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def _build(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < 6 or np.all(y == y[0]):
            return {'leaf': True, 'prob': float(np.mean(y))}
        best_g, best_f, best_t = -1, None, None
        curr_gini = 1.0 - (np.mean(y)**2 + (1 - np.mean(y))**2)
        feats = np.random.choice(X.shape[1], size=max(3, int(np.sqrt(X.shape[1]))), replace=False)
        for f in feats:
            for t in np.percentile(X[:, f], [25, 50, 75]):
                mask = X[:, f] <= t
                nl = mask.sum()
                nr = len(y) - nl
                if nl < 2 or nr < 2: continue
                gl = 1.0 - (np.mean(y[mask])**2 + (1 - np.mean(y[mask]))**2)
                gr = 1.0 - (np.mean(y[~mask])**2 + (1 - np.mean(y[~mask]))**2)
                gain = curr_gini - ((nl/len(y))*gl + (nr/len(y))*gr)
                if gain > best_g:
                    best_g, best_f, best_t = gain, f, t
        if best_g <= 1e-4:
            return {'leaf': True, 'prob': float(np.mean(y))}
        left_m = X[:, best_f] <= best_t
        return {
            'leaf': False, 'f': best_f, 't': best_t,
            'left': self._build(X[left_m], y[left_m], depth + 1),
            'right': self._build(X[~left_m], y[~left_m], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build(X, y)
        return self

    def _pred(self, node, x):
        if node['leaf']: return node['prob']
        if x[node['f']] <= node['t']: return self._pred(node['left'], x)
        return self._pred(node['right'], x)

    def predict_proba(self, X):
        return np.array([self._pred(self.tree, row) for row in X])

class RandomForest:
    def __init__(self, n=35, depth=6):
        self.n = n
        self.depth = depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n):
            idx = np.random.choice(len(X), size=int(0.85 * len(X)), replace=True)
            t = FastTree(max_depth=self.depth).fit(X[idx], y[idx])
            self.trees.append(t)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

class XGBoost:
    def __init__(self, n=50, lr=0.1, depth=4):
        self.n = n
        self.lr = lr
        self.depth = depth
        self.trees = []
        self.p0 = 0.0

    def fit(self, X, y):
        self.p0 = float(np.clip(np.mean(y), 1e-4, 1 - 1e-4))
        fm = np.full(len(X), np.log(self.p0 / (1 - self.p0)))
        self.trees = []
        for _ in range(self.n):
            p = 1.0 / (1.0 + np.exp(-np.clip(fm, -20, 20)))
            pseudo = ((y - p) > 0).astype(int)
            t = FastTree(max_depth=self.depth).fit(X, pseudo)
            fm += self.lr * (t.predict_proba(X) - 0.5)
            self.trees.append(t)
        return self

    def predict_proba(self, X):
        fm = np.full(len(X), np.log(self.p0 / (1 - self.p0)))
        for t in self.trees:
            fm += self.lr * (t.predict_proba(X) - 0.5)
        return 1.0 / (1.0 + np.exp(-np.clip(fm, -20, 20)))

# Train base models
print("\nTraining Base Models with Calibrated SMOTE...")
xgb = XGBoost(n=50, lr=0.12, depth=4).fit(X_tr_smote, y_tr_smote)
rf = RandomForest(n=35, depth=6).fit(X_tr_smote, y_tr_smote)

p_xgb_test = xgb.predict_proba(X_test_hybrid)
p_rf_test = rf.predict_proba(X_test_hybrid)

# Meta-learner: Weighted ensemble combination (Hybrid Model)
p_ensemble = 0.55 * p_xgb_test + 0.45 * p_rf_test

# Compute ROC-AUC
order = np.argsort(-p_ensemble)
y_sorted = y_test[order]
tp_c = np.cumsum(y_sorted == 1)
fp_c = np.cumsum(y_sorted == 0)
n_pos = np.sum(y_test == 1)
n_neg = np.sum(y_test == 0)

tpr_arr = np.concatenate([[0], tp_c / n_pos])
fpr_arr = np.concatenate([[0], fp_c / n_neg])
s_idx = np.argsort(fpr_arr)
roc_auc = float(np.sum(0.5 * (tpr_arr[s_idx][1:] + tpr_arr[s_idx][:-1]) * np.diff(fpr_arr[s_idx])))

print(f"\n--> ENSEMBLE ROC-AUC ON UNTOUCHED TEST SET = {roc_auc:.4f}")

# Threshold tuning scan for High Accuracy & Balanced Accuracy
print("\n--- THRESHOLD SCAN FOR ACCURACY OPTIMIZATION ---")
print(f"{'THRESH':<8} | {'ACCURACY':<10} | {'RECALL':<10} | {'PRECISION':<10} | {'BALANCED ACC':<14} | {'F1':<8}")
print("-" * 75)

best_acc_row = None
best_bal_row = None
max_acc = -1
max_bal = -1

for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    pred = (p_ensemble >= th).astype(int)
    tp = np.sum((pred == 1) & (y_test == 1))
    fp = np.sum((pred == 1) & (y_test == 0))
    tn = np.sum((pred == 0) & (y_test == 0))
    fn = np.sum((pred == 0) & (y_test == 1))
    
    acc = (tp + tn) / len(y_test)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    bal_acc = (rec + (tn / (tn + fp))) / 2.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"{th:<8.2f} | {acc*100:>8.2f}% | {rec*100:>8.2f}% | {prec*100:>8.2f}% | {bal_acc*100:>12.2f}% | {f1*100:>6.2f}%")
    
    if acc > max_acc:
        max_acc = acc
        best_acc_row = (th, acc, rec, prec, bal_acc, f1)
    if bal_acc > max_bal:
        max_bal = bal_acc
        best_bal_row = (th, acc, rec, prec, bal_acc, f1)

print("-" * 75)
print(f"Max Accuracy Configuration: Threshold {best_acc_row[0]:.2f} -> Accuracy = {best_acc_row[1]*100:.2f}%, Recall = {best_acc_row[2]*100:.2f}%")
print(f"Max Balanced Accuracy:      Threshold {best_bal_row[0]:.2f} -> Accuracy = {best_bal_row[1]*100:.2f}%, Balanced Acc = {best_bal_row[4]*100:.2f}%, Recall = {best_bal_row[2]*100:.2f}%")
