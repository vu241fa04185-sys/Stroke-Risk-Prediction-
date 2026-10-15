import numpy as np
from check_train_accuracy import X_tr_smote, y_tr_smote, X_test_hybrid, y_test

# 1. Decision Tree (unconstrained/deep)
from check_train_accuracy import FastTree

class DeepTree:
    def __init__(self, max_depth=16, min_samples=2):
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
            for t in np.percentile(X[:, f], [10, 25, 40, 50, 60, 75, 90]):
                mask = X[:, f] <= t
                nl, nr = mask.sum(), len(y) - mask.sum()
                if nl < 1 or nr < 1: continue
                gain = curr_g - ((nl/len(y))*(1 - np.mean(y[mask])**2 - (1-np.mean(y[mask]))**2) + (nr/len(y))*(1 - np.mean(y[~mask])**2 - (1-np.mean(y[~mask]))**2))
                if gain > best_g:
                    best_g, best_f, best_t = gain, f, t
        if best_g <= 1e-5:
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

dt = DeepTree(max_depth=16, min_samples=2).fit(X_tr_smote, y_tr_smote)
p_tr_dt = dt.predict_proba(X_tr_smote)
acc_tr_dt = np.mean((p_tr_dt >= 0.5) == y_tr_smote)
print(f"Deep Decision Tree Training Accuracy: {acc_tr_dt*100:.2f}%")

# Random Forest with Deep Trees
class DeepRF:
    def __init__(self, n=40, depth=14):
        self.n, self.depth = n, depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = len(X)
        for _ in range(self.n):
            idx = np.random.choice(n_samples, size=int(0.9 * n_samples), replace=True)
            t = DeepTree(max_depth=self.depth, min_samples=2).fit(X[idx], y[idx])
            self.trees.append(t)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

rf = DeepRF(n=40, depth=14).fit(X_tr_smote, y_tr_smote)
p_tr_rf = rf.predict_proba(X_tr_smote)
acc_tr_rf = np.mean((p_tr_rf >= 0.5) == y_tr_smote)
print(f"Deep Random Forest Training Accuracy: {acc_tr_rf*100:.2f}%")
