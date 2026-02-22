"""
Pertemuan 2: Data Preprocessing dan EDA
=======================================
Contoh pengerjaan lengkap sesuai materi pertemuan-02.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ============================================================
# Memuat dataset Titanic
# ============================================================
df = sns.load_dataset("titanic")
print("=== Ukuran Data Awal ===")
print("Shape:", df.shape)
print("\nTipe kolom:")
print(df.dtypes)

# ============================================================
# 1. Missing Value
# ============================================================
print("\n=== Missing Value (sebelum) ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
df["deck"] = df["deck"].cat.add_categories("Unknown").fillna("Unknown")

print("\n=== Missing Value (sesudah) ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ============================================================
# 2. Duplikat
# ============================================================
print("\n=== Duplikat ===")
print("Jumlah duplikat:", df.duplicated().sum())
df = df.drop_duplicates()

# ============================================================
# 3. Outlier (IQR) pada kolom fare dan age
# ============================================================
print("\n=== Outlier (IQR) ===")
for col in ["fare", "age"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers_before = ((df[col] < low) | (df[col] > high)).sum()
    df[col] = df[col].clip(low, high)
    print(f"  {col}: {outliers_before} outlier dipotong (batas {low:.1f} – {high:.1f})")

# ============================================================
# 4. Scaling
# ============================================================
scaler = StandardScaler()
df[["age", "fare"]] = scaler.fit_transform(df[["age", "fare"]])

print("\n=== Statistik Setelah Scaling ===")
print(df[["age", "fare"]].describe().round(3))

# ============================================================
# 5. EDA — 5 Visualisasi + 5 Insight
# ============================================================

# Visualisasi 1: distribusi target (survived)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

df["survived"].value_counts().plot(kind="bar", ax=axes[0, 0], color=["salmon", "skyblue"])
axes[0, 0].set_title("1. Distribusi Survived")
axes[0, 0].set_xticklabels(["Tidak", "Ya"], rotation=0)

# Visualisasi 2: distribusi umur
axes[0, 1].hist(df["age"], bins=20, color="steelblue", edgecolor="black")
axes[0, 1].set_title("2. Distribusi Age (scaled)")

# Visualisasi 3: distribusi fare
axes[0, 2].hist(df["fare"], bins=20, color="coral", edgecolor="black")
axes[0, 2].set_title("3. Distribusi Fare (scaled)")

# Visualisasi 4: sex vs survived
df.groupby("sex")["survived"].mean().plot(kind="bar", ax=axes[1, 0], color="mediumpurple")
axes[1, 0].set_title("4. Survival Rate by Sex")
axes[1, 0].set_xticklabels(["female", "male"], rotation=0)

# Visualisasi 5: class vs survived
df.groupby("class")["survived"].mean().plot(kind="bar", ax=axes[1, 1], color="teal")
axes[1, 1].set_title("5. Survival Rate by Class")
axes[1, 1].set_xticklabels(["First", "Second", "Third"], rotation=0)

# Kosongkan subplot terakhir
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("pertemuan02_eda.png")
plt.close()
print("\nVisualisasi disimpan ke pertemuan02_eda.png")

# 5 Insight
print("\n=== 5 Insight ===")
surv_rate = df["survived"].mean()
print(f"1. Tingkat survival keseluruhan: {surv_rate:.1%}")
print(f"2. Survival rate perempuan ({df[df['sex']=='female']['survived'].mean():.1%}) "
      f"jauh lebih tinggi daripada laki-laki ({df[df['sex']=='male']['survived'].mean():.1%}).")
print(f"3. Penumpang kelas 1 memiliki survival rate tertinggi "
      f"({df[df['class']=='First']['survived'].mean():.1%}).")
print(f"4. Median age setelah scaling mendekati 0 (mean={df['age'].mean():.3f}), sesuai harapan StandardScaler.")
print(f"5. Setelah clipping, fare tidak lagi memiliki outlier ekstrem.")

print("\n✅ Pertemuan 02 selesai.")
