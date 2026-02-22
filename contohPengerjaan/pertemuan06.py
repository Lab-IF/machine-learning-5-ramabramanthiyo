"""
Pertemuan 6: Ensemble Methods (Random Forest & Gradient Boosting)
================================================================
Contoh pengerjaan lengkap sesuai materi pertemuan-06.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# Tugas 1: Bandingkan DT vs RF vs GB (Classification)
# ============================================================
print("=" * 55)
print("Tugas 1: Perbandingan DT vs RF vs GB (Classification)")
print("=" * 55)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, pred), 3)
    results[name] = acc
    print(f"  {name:20s} accuracy: {acc}")

# ============================================================
# Tugas 2: Top 10 Feature Importance dari Random Forest
# ============================================================
print("\n" + "=" * 55)
print("Tugas 2: Top 10 Feature Importance (Random Forest)")
print("=" * 55)

rf = models["RandomForest"]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

for rank, idx in enumerate(indices, 1):
    print(f"  {rank:2d}. {data.feature_names[idx]:30s} importance: {importances[idx]:.4f}")

# ============================================================
# Tugas 3: Ubah n_estimators dan lihat pengaruhnya
# ============================================================
print("\n" + "=" * 55)
print("Tugas 3: Pengaruh n_estimators pada Random Forest")
print("=" * 55)

for n in [10, 50, 100, 200, 500]:
    rf_n = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_n.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf_n.predict(X_test))
    print(f"  n_estimators={n:>3d} → accuracy: {acc:.3f}")

# ============================================================
# Bonus: Ensemble Regression (sesuai materi)
# ============================================================
print("\n" + "=" * 55)
print("Bonus: Ensemble Regression (Random Forest)")
print("=" * 55)

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

db = load_diabetes()
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    db.data, db.target, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_r, y_train_r)
pred_r = reg.predict(X_test_r)
print("RMSE:", round(root_mean_squared_error(y_test_r, pred_r), 3))
print("R2  :", round(r2_score(y_test_r, pred_r), 3))

print("\n✅ Pertemuan 06 selesai.")
