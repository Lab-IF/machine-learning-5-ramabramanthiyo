"""
Pertemuan 1: Dasar Python untuk Machine Learning
=================================================
Contoh pengerjaan lengkap sesuai materi pertemuan-01.
"""

import matplotlib
matplotlib.use("Agg")  # non-GUI backend agar bisa jalan tanpa display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ============================================================
# 1. NumPy — operasi dasar array
# ============================================================
x = np.array([1, 2, 3, 4, 5])
print("=== NumPy ===")
print("mean:", x.mean())

# ============================================================
# 2. Pandas — DataFrame sederhana
# ============================================================
df = pd.DataFrame({"nama": ["A", "B", "C"], "nilai": [80, 75, 90]})
print("\n=== Pandas ===")
print(df)

# ============================================================
# 3. Visualisasi — bar chart
# ============================================================
df.plot(kind="bar", x="nama", y="nilai", legend=False)
plt.title("Nilai Mahasiswa")
plt.tight_layout()
plt.savefig("pertemuan01_barchart.png")
plt.close()
print("\n=== Visualisasi ===")
print("Bar chart disimpan ke pertemuan01_barchart.png")

# ============================================================
# 4. Dataset ML — load Iris
# ============================================================
iris = load_iris(as_frame=True)
print("\n=== Dataset Iris (5 baris pertama) ===")
print(iris.frame.head())

# ============================================================
# TUGAS: contoh pengerjaan
# ============================================================

# Tugas 1: array random 20 angka, hitung mean, median, std
print("\n=== Tugas 1: Statistik Array Random ===")
arr = np.random.seed(42)
arr = np.random.rand(20) * 100
print("Array:", arr.round(1))
print("Mean :", round(arr.mean(), 2))
print("Median:", round(np.median(arr), 2))
print("Std  :", round(arr.std(), 2))

# Tugas 2: DataFrame 10 mahasiswa, cari 3 nilai tertinggi
print("\n=== Tugas 2: Top 3 Nilai ===")
np.random.seed(0)
df_mhs = pd.DataFrame({
    "nama": [f"Mahasiswa_{i+1}" for i in range(10)],
    "nilai": np.random.randint(50, 100, 10)
})
print(df_mhs)
top3 = df_mhs.nlargest(3, "nilai")
print("\nTop 3:")
print(top3)

# Tugas 3: 2 grafik dari data nilai
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(df_mhs["nama"], df_mhs["nilai"], color="steelblue")
axes[0].set_title("Bar Chart Nilai")
axes[0].tick_params(axis="x", rotation=45)

axes[1].hist(df_mhs["nilai"], bins=5, color="coral", edgecolor="black")
axes[1].set_title("Histogram Nilai")

plt.tight_layout()
plt.savefig("pertemuan01_tugas_grafik.png")
plt.close()
print("\n=== Tugas 3 ===")
print("Grafik disimpan ke pertemuan01_tugas_grafik.png")

# Tugas 4: Load dataset load_wine, tulis 3 insight
from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)
print("\n=== Tugas 4: Dataset Wine ===")
print("Ukuran data:", wine.frame.shape)
print("Kolom:", list(wine.frame.columns))
print("\nInsight 1: Dataset Wine memiliki", wine.frame.shape[0], "sampel dan",
      wine.frame.shape[1] - 1, "fitur.")
print("Insight 2: Target memiliki", wine.frame["target"].nunique(), "kelas:",
      sorted(wine.frame["target"].unique()))
print("Insight 3: Rata-rata alcohol =", round(wine.frame["alcohol"].mean(), 2))

print("\n✅ Pertemuan 01 selesai.")
