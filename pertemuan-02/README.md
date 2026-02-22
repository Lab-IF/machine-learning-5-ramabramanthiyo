# Pertemuan 2: Data Preprocessing dan EDA

## Tujuan
Peserta mampu:
- Membersihkan data (missing value, duplikat, outlier).
- Melakukan scaling (normalisasi/standardisasi).
- Membaca pola data lewat EDA.

## Konsep Inti
- Kualitas data sangat menentukan kualitas model.
- EDA dipakai untuk memahami data sebelum modeling.
- Preprocessing umum: bersihkan data -> ubah format -> scaling -> split data.

## Contoh Kode Ringkas

**Tujuan:** Menunjukkan 4 langkah preprocessing data yang paling umum — menangani missing value, menghapus duplikat, memotong outlier, dan melakukan scaling — agar data siap dipakai model ML.
**Kapan dipakai:** Selalu dilakukan **sebelum** melatih model, di setiap project ML.

```python
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Memuat dataset Titanic bawaan seaborn (891 penumpang, 15 kolom).
df = sns.load_dataset("titanic")

# 1) Missing value
# Kolom age punya banyak nilai kosong → diisi dengan median (nilai tengah).
# Kolom embarked (pelabuhan) → diisi dengan modus (nilai paling sering).
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# 2) Duplikat
# Menghapus baris yang isinya 100% sama persis.
df = df.drop_duplicates()

# 3) Outlier sederhana (IQR) pada fare
# Menghitung batas bawah dan atas. Nilai di luar batas dipotong (clip).
q1, q3 = df["fare"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
df["fare"] = df["fare"].clip(low, high)

# 4) Scaling
# StandardScaler mengubah data agar mean=0, std=1.
# Penting untuk algoritma yang sensitif terhadap skala (SVM, KNN, dll).
scaler = StandardScaler()
df[["age", "fare"]] = scaler.fit_transform(df[["age", "fare"]])

print(df[["age", "fare"]].describe())
```

**Kode opsional:**
- Metode pengisian missing value bisa diganti: `fillna(mean())` (rata-rata), `fillna(0)` (nol), atau pakai `SimpleImputer` dari sklearn.
- `clip(low, high)` bisa diganti dengan **menghapus** baris outlier: `df = df[(df["fare"] >= low) & (df["fare"] <= high)]`.
- `StandardScaler` bisa diganti `MinMaxScaler` jika ingin data di-range 0–1.

## EDA Minimal yang Wajib
1. Distribusi target (contoh: `survived`).
2. 2 grafik univariate (misalnya umur, fare).
3. 2 grafik bivariate (misalnya `sex` vs `survived`, `class` vs `survived`).
4. Heatmap korelasi fitur numerik.

## Tugas
Gunakan satu dataset (bebas, boleh Titanic):
1. Tampilkan ukuran data dan tipe kolom.
2. Tangani missing value dengan alasan yang jelas.
3. Tangani outlier minimal pada 2 kolom numerik.
4. Lakukan scaling pada kolom numerik.
5. Buat minimal 5 visualisasi + 5 insight.

## Output Pengumpulan
- Notebook: `NIM_Nama_Pertemuan02.ipynb`
- Setiap langkah diberi markdown singkat (apa dan kenapa).
