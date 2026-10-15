import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Load Data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
if 'id' in df.columns: df.drop(columns=['id'], inplace=True)
df = df[df['gender'] != 'Other'].copy()
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# Spline Imputation
df.sort_values(by='age', inplace=True)
known_mask = ~df['bmi'].isna()
ages = df.loc[known_mask, 'age'].values
bmis = df.loc[known_mask, 'bmi'].values
u_ages, u_idx = np.unique(ages, return_index=True)
spline = CubicSpline(u_ages, bmis[u_idx], extrapolate=True)
df.loc[~known_mask, 'bmi'] = np.clip(spline(df.loc[~known_mask, 'age']), 12, 65)

# Encode
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

# 80/20 Split
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
np.random.seed(42)
np.random.shuffle(pos_idx)
np.random.shuffle(neg_idx)

n_test_pos = int(np.round(0.20 * len(pos_idx)))
n_test_neg = int(np.round(0.20 * len(neg_idx)))

test_idx = np.concatenate([pos_idx[:n_test_pos], neg_idx[:n_test_neg]])
train_idx = np.concatenate([pos_idx[n_test_pos:], neg_idx[n_test_neg:]])

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Scaling & PCA
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

U, S, Vt = np.linalg.svd(X_train_scaled, full_matrices=False)
W_pca = Vt[:8].T

X_train_hybrid = np.hstack([X_train_scaled, np.dot(X_train_scaled, W_pca)])
X_test_hybrid = np.hstack([X_test_scaled, np.dot(X_test_scaled, W_pca)])

# SMOTE on train (1:1 balance)
pos = np.where(y_train == 1)[0]
neg = np.where(y_train == 0)[0]
needed = len(neg) - len(pos)
X_p = X_train_hybrid[pos]
synthetic = []
for _ in range(needed):
    i = np.random.choice(len(pos))
    dists = np.sum((X_p - X_p[i]) ** 2, axis=1)
    knn = np.argsort(dists)[1:min(6, len(pos))]
    nb = np.random.choice(knn)
    synthetic.append(X_p[i] + np.random.rand() * (X_p[nb] - X_p[i]))

X_tr_smote = np.vstack([X_train_hybrid, np.array(synthetic)])
y_tr_smote = np.concatenate([y_train, np.ones(needed, dtype=int)])

# Fast Tree
class FastTree:
    def __init__(self, max_depth=8, min_samples=4):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def _build(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples or np.all(y == y[0]):
            return {'leaf': True, 'prob': float(np.mean(y))}
        best_g, best_f, best_t = -1, None, None
        curr_g = 1.0 - (np.mean(y)**2 + (1 - np.mean(y))**2)
        feats = np.random.choice(X.shape[1], size=min(X.shape[1], max(3, int(np.sqrt(X.shape[1])))), replace=False)
        for f in feats:
            for t in np.percentile(X[:, f], [15, 30, 50, 70, 85]):
                mask = X[:, f] <= t
                nl, nr = mask.sum(), len(y) - mask.sum()
                if nl < 2 or nr < 2: continue
                gain = curr_g - ((nl/len(y))*(1 - np.mean(y[mask])**2 - (1-np.mean(y[mask]))**2) + (nr/len(y))*(1 - np.mean(y[~mask])**2 - (1-np.mean(y[~mask]))**2))
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
    def __init__(self, n=60, depth=12):
        self.n, self.depth = n, depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = len(X)
        for _ in range(self.n):
            idx = np.random.choice(n_samples, size=int(0.85 * n_samples), replace=True)
            t = FastTree(max_depth=self.depth).fit(X[idx], y[idx])
            self.trees.append(t)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

print("Training Random Forest on SMOTE training dataset...")
rf = RandomForest(n=50, depth=9).fit(X_tr_smote, y_tr_smote)

p_train = rf.predict_proba(X_tr_smote)
train_acc = np.mean((p_train >= 0.5) == y_tr_smote)
print(f"--> SMOTE Training Dataset Accuracy: {train_acc*100:.2f}%")

p_raw_train = rf.predict_proba(X_train_hybrid)
raw_train_acc = np.mean((p_raw_train >= 0.5) == y_train)
print(f"--> Original Training Dataset Accuracy: {raw_train_acc*100:.2f}%")
