"""
Pertemuan 8: UTS Mini Project ML (End-to-End)
=============================================
Contoh pengerjaan lengkap sesuai materi pertemuan-08.
Menggunakan dataset Breast Cancer bawaan sklearn (Classification).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================================
# 1) Problem & Dataset
# ==========================================================
print("=" * 60)
print("1) PROBLEM & DATASET")
print("=" * 60)
print("Masalah : Klasifikasi tumor payudara (malignant / benign)")
print("Tujuan  : Memprediksi apakah tumor ganas atau jinak")
print("Sumber  : sklearn.datasets.load_breast_cancer")

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malignant, 1 = benign

print(f"\nUkuran data  : {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"Fitur        : {len(data.feature_names)} fitur numerik")
print(f"Target       : {list(data.target_names)} (0=malignant, 1=benign)")
print(f"Distribusi target:")
print(df["target"].value_counts().rename({0: "malignant", 1: "benign"}))

# ==========================================================
# 2) EDA
# ==========================================================
print("\n" + "=" * 60)
print("2) EDA")
print("=" * 60)

# Cek missing value
print("\nMissing values:", df.isnull().sum().sum())

# Statistik deskriptif
print("\nStatistik deskriptif (5 fitur pertama):")
print(df.iloc[:, :5].describe().round(2))

# 5 Visualisasi
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Viz 1: Distribusi target
df["target"].value_counts().plot(kind="bar", ax=axes[0, 0], color=["salmon", "skyblue"])
axes[0, 0].set_title("1. Distribusi Target")
axes[0, 0].set_xticklabels(["Malignant", "Benign"], rotation=0)

# Viz 2: Histogram mean radius
axes[0, 1].hist(df["mean radius"], bins=20, color="steelblue", edgecolor="black")
axes[0, 1].set_title("2. Distribusi Mean Radius")

# Viz 3: Histogram mean texture
axes[0, 2].hist(df["mean texture"], bins=20, color="coral", edgecolor="black")
axes[0, 2].set_title("3. Distribusi Mean Texture")

# Viz 4: Boxplot mean radius per target
df.boxplot(column="mean radius", by="target", ax=axes[1, 0])
axes[1, 0].set_title("4. Mean Radius by Target")
axes[1, 0].set_xlabel("Target (0=malignant, 1=benign)")

# Viz 5: Korelasi heatmap (top 6 fitur)
top_cols = ["mean radius", "mean texture", "mean perimeter", "mean area",
            "mean smoothness", "target"]
sns.heatmap(df[top_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", ax=axes[1, 1])
axes[1, 1].set_title("5. Heatmap Korelasi")

axes[1, 2].axis("off")

plt.suptitle("")
plt.tight_layout()
plt.savefig("pertemuan08_eda.png")
plt.close()
print("Visualisasi EDA disimpan ke pertemuan08_eda.png")

# 5 Insight
print("\n5 Insight:")
benign_pct = (df["target"] == 1).mean() * 100
print(f"  1. Dataset sedikit imbalanced: {benign_pct:.1f}% benign, {100-benign_pct:.1f}% malignant.")
print(f"  2. Mean radius tumor malignant (avg {df[df['target']==0]['mean radius'].mean():.1f}) "
      f"lebih besar dari benign (avg {df[df['target']==1]['mean radius'].mean():.1f}).")
print(f"  3. Mean perimeter sangat berkorelasi tinggi dengan mean radius dan mean area.")
print(f"  4. Tidak ada missing value dalam dataset.")
print(f"  5. Fitur 'mean smoothness' memiliki korelasi rendah dengan target, "
      f"mungkin kurang informatif.")

# ==========================================================
# 3) Preprocessing
# ==========================================================
print("\n" + "=" * 60)
print("3) PREPROCESSING")
print("=" * 60)

X = df.drop(columns=["target"])
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} sampel | Test: {X_test.shape[0]} sampel")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("StandardScaler diterapkan pada data latih dan uji.")
print("Alasan: beberapa model (Logistic Regression) sensitif terhadap skala fitur.")

# ==========================================================
# 4) Modeling — minimal 3 model
# ==========================================================
print("\n" + "=" * 60)
print("4) MODELING")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    results[name] = acc

# Tabel perbandingan
print("\nTabel Perbandingan:")
print(f"{'Model':25s}  {'Accuracy':>8s}")
print("-" * 36)
for name, acc in results.items():
    print(f"{name:25s}  {acc:>8.3f}")

# Detail model terbaik
best_name = max(results, key=results.get)
best_model = models[best_name]
pred_best = best_model.predict(X_test_scaled)
print(f"\nDetail model terbaik ({best_name}):")
print(confusion_matrix(y_test, pred_best))
print(classification_report(y_test, pred_best, target_names=data.target_names))

# ==========================================================
# 5) Tuning (GridSearchCV pada model terbaik)
# ==========================================================
print("=" * 60)
print("5) TUNING")
print("=" * 60)

if "Random Forest" in best_name:
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
elif "Logistic" in best_name:
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=500, random_state=42),
        param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
else:
    param_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )

grid.fit(X_train_scaled, y_train)
print(f"Model yang di-tune: {best_name}")
print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.3f}")

tuned_pred = grid.predict(X_test_scaled)
tuned_acc = accuracy_score(y_test, tuned_pred)
print(f"Test accuracy setelah tuning: {tuned_acc:.3f}")

# ==========================================================
# 6) Kesimpulan
# ==========================================================
print("\n" + "=" * 60)
print("6) KESIMPULAN")
print("=" * 60)
print(f"• Model terbaik: {best_name} (accuracy sebelum tuning: {results[best_name]:.3f})")
print(f"• Setelah tuning dengan GridSearchCV: accuracy {tuned_acc:.3f}")
print(f"• Best parameters: {grid.best_params_}")
print("• Dataset Breast Cancer sudah cukup bersih (tanpa missing value),")
print("  sehingga preprocessing hanya perlu scaling.")
print("• Rekomendasi: coba tambahkan feature selection atau gunakan")
print("  model lain (SVM, Gradient Boosting) untuk memperbaiki hasil.")

print("\n✅ Pertemuan 08 (UTS Mini Project) selesai.")
