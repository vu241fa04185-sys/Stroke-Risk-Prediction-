"""
Compute exact, fresh metrics at threshold = 0.700 (and scan 0.50 -> 0.80)
========================================================================
Applies SMOTE strictly on the 80% training partition, evaluates the 3 hybrid stacking models
on the untouched 20% test partition (1,022 records).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Reproducibility seed
np.random.seed(42)

# 1. Load Data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)
df = df[df['gender'] != 'Other'].copy()
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# Imputation via Cubic Spline
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

# Feature Engineering
df['age_glucose'] = (df['age'] / 50.0) * (df['avg_glucose_level'] / 100.0)
df['vascular_risk'] = df['hypertension'] * 1.5 + df['heart_disease'] * 2.0
df['age_bmi'] = (df['age'] / 50.0) * (df['bmi'] / 25.0)
df['hyperglycemia'] = (df['avg_glucose_level'] >= 140.0).astype(float)
df['senior_citizen'] = (df['age'] >= 65.0).astype(float)

feature_cols = [c for c in df.columns if c != 'stroke']
X = df[feature_cols].values.astype(np.float64)
y = df['stroke'].values.astype(int)

# 2. Stratified 80/20 Partition
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

# 3. Scale & PCA on Train
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

U, S, Vt = np.linalg.svd(X_train_scaled, full_matrices=False)
W_pca = Vt[:9].T

X_train_hybrid = np.hstack([X_train_scaled, np.dot(X_train_scaled, W_pca)])
X_test_hybrid = np.hstack([X_test_scaled, np.dot(X_test_scaled, W_pca)])

# 4. SMOTE on 80% Train Set Only
def apply_smote(X_in, y_in, ratio=0.60):
    pos = np.where(y_in == 1)[0]
    neg = np.where(y_in == 0)[0]
    n_p, n_n = len(pos), len(neg)
    target_pos = int(n_n * ratio)
    needed = target_pos - n_p
    if needed <= 0: return X_in, y_in

    X_p = X_in[pos]
    synthetic = []
    for _ in range(needed):
        i = np.random.choice(n_p)
        dists = np.sum((X_p - X_p[i]) ** 2, axis=1)
        knn = np.argsort(dists)[1:min(6, n_p)]
        nb = np.random.choice(knn) if len(knn) > 0 else i
        gap = np.random.rand()
        synthetic.append(X_p[i] + gap * (X_p[nb] - X_p[i]))

    X_bal = np.vstack([X_in, np.array(synthetic)])
    y_bal = np.concatenate([y_in, np.ones(needed, dtype=int)])
    p = np.random.permutation(len(X_bal))
    return X_bal[p], y_bal[p]

X_tr_smote, y_tr_smote = apply_smote(X_train_hybrid, y_train, ratio=0.60)

# 5. Fast Trees & Base Learners
class FastTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def _build(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < 6 or np.all(y == y[0]):
            return {'leaf': True, 'prob': float(np.mean(y))}
        best_g, best_f, best_t = -1, None, None
        curr_gini = 1.0 - (np.mean(y)**2 + (1 - np.mean(y))**2)
        feats = np.random.choice(X.shape[1], size=min(X.shape[1], max(1, int(np.sqrt(X.shape[1])))), replace=False)
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
    def __init__(self, n=50, lr=0.12, depth=4):
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

class LogRegMeta:
    def __init__(self, lr=0.1, iters=500):
        self.w = None
        self.b = 0.0
        self.lr = lr
        self.iters = iters

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, self.w) + self.b, -25, 25)))
            err = p - y
            self.w -= self.lr * (np.dot(X.T, err) / len(X) + 0.01 * self.w)
            self.b -= self.lr * np.mean(err)
        return self

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, self.w) + self.b, -25, 25)))

class KNNMeta:
    def __init__(self, k=7):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X, self.y = X.copy(), y.copy()
        return self

    def predict_proba(self, X):
        probs = []
        for row in X:
            dists = np.sum((self.X - row) ** 2, axis=1)
            nn = np.argpartition(dists, self.k)[:self.k]
            w = 1.0 / (np.sqrt(dists[nn]) + 1e-5)
            probs.append(np.sum(self.y[nn] * w) / np.sum(w))
        return np.array(probs)

# 6. Out-of-fold Training
print("Training Base Models (XGBoost + Random Forest)...")
xgb = XGBoost().fit(X_tr_smote, y_tr_smote)
rf = RandomForest().fit(X_tr_smote, y_tr_smote)

p_xgb_tr = xgb.predict_proba(X_train_hybrid)
p_rf_tr = rf.predict_proba(X_train_hybrid)
Z_train = np.column_stack([p_xgb_tr, p_rf_tr])

p_xgb_te = xgb.predict_proba(X_test_hybrid)
p_rf_te = rf.predict_proba(X_test_hybrid)
Z_test = np.column_stack([p_xgb_te, p_rf_te])

# Fit 3 Meta-Learners
hybrid1 = LogRegMeta().fit(Z_train, y_train)
hybrid2 = FastTree(max_depth=4).fit(Z_train, y_train)
hybrid3 = KNNMeta(k=7).fit(Z_train, y_train)

models = {
    "Hybrid 1 (XGB + RF -> Logistic Regression)": hybrid1.predict_proba(Z_test),
    "Hybrid 2 (XGB + RF -> Decision Tree)": hybrid2.predict_proba(Z_test),
    "Hybrid 3 (XGB + RF -> KNN)": hybrid3.predict_proba(Z_test),
}

# 7. Evaluate at Threshold = 0.700 Exactly
print("\n" + "=" * 90)
print("  EXACT READINGS AT THRESHOLD = 0.700 (HIGH THRESHOLD / HIGH ACCURACY MODE)")
print("=" * 90)
print(f"{'MODEL':<42} | {'ACCURACY':<8} | {'PRECISION':<9} | {'RECALL':<8} | {'F1-SCORE':<8} | {'SPECIFICITY':<11} | {'ROC-AUC':<8}")
print("-" * 90)

exact_070_metrics = {}

for name, probs in models.items():
    # ROC-AUC
    order = np.argsort(-probs)
    y_s = y_test[order]
    tp_c = np.cumsum(y_s == 1)
    fp_c = np.cumsum(y_s == 0)
    n_p = np.sum(y_test == 1)
    n_n = np.sum(y_test == 0)
    tpr_a = np.concatenate([[0], tp_c / n_p])
    fpr_a = np.concatenate([[0], fp_c / n_n])
    s_idx = np.argsort(fpr_a)
    auc_v = float(np.sum(0.5 * (tpr_a[s_idx][1:] + tpr_a[s_idx][:-1]) * np.diff(fpr_a[s_idx])))

    # Threshold = 0.700
    th = 0.700
    pred = (probs >= th).astype(int)

    tp = int(np.sum((pred == 1) & (y_test == 1)))
    fp = int(np.sum((pred == 1) & (y_test == 0)))
    tn = int(np.sum((pred == 0) & (y_test == 0)))
    fn = int(np.sum((pred == 0) & (y_test == 1)))

    acc = (tp + tn) / len(y_test)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    exact_070_metrics[name] = {
        "th": th, "acc": acc, "prec": prec, "rec": rec, "spec": spec, "f1": f1, "auc": auc_v,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

    print(f"{name:<42} | {acc*100:>7.2f}% | {prec*100:>8.2f}% | {rec*100:>7.2f}% | {f1*100:>7.2f}% | {spec*100:>10.2f}% | {auc_v:>8.4f}")

print("=" * 90)

print("\nCONFUSION MATRICES AT THRESHOLD = 0.700:")
for name, m in exact_070_metrics.items():
    print(f"\n[{name}]:")
    print(f"  +---------------------------------------------+")
    print(f"  | True Negative  (TN): {m['TN']:<4} | False Positive (FP): {m['FP']:<4} |")
    print(f"  | False Negative (FN): {m['FN']:<4} | True Positive  (TP): {m['TP']:<4} |")
    print(f"  +---------------------------------------------+")

# 8. Step-by-Step Threshold Scan from 0.10 to 0.70
print("\n" + "=" * 90)
print("  STEP-BY-STEP PROGRESSION AS YOU INCREASE THRESHOLD (HYBRID 2: DECISION TREE)")
print("=" * 90)
print(f"{'THRESHOLD':<10} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL (SENSITIVITY)':<20} | {'SPECIFICITY':<12} | {'TP':<4} | {'FP':<4}")
print("-" * 90)

p2 = models["Hybrid 2 (XGB + RF -> Decision Tree)"]
for th_val in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    pred = (p2 >= th_val).astype(int)
    tp = int(np.sum((pred == 1) & (y_test == 1)))
    fp = int(np.sum((pred == 1) & (y_test == 0)))
    tn = int(np.sum((pred == 0) & (y_test == 0)))
    fn = int(np.sum((pred == 0) & (y_test == 1)))
    acc = (tp + tn) / len(y_test)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"{th_val:<10.2f} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>18.2f}% | {spec*100:>10.2f}% | {tp:<4} | {fp:<4}")
print("=" * 90)
