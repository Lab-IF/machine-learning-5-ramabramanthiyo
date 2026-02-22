# Pertemuan 1: Dasar Python untuk Machine Learning

## Tujuan
Setelah pertemuan ini, peserta bisa:
- Menyiapkan environment Python untuk praktikum ML.
- Memakai `NumPy`, `Pandas`, `Matplotlib`, dan dataset bawaan `scikit-learn`.
- Menjalankan analisis data sederhana di Jupyter Notebook.

## Konsep Inti (Sangat Singkat)
- `NumPy` untuk angka dan array.
- `Pandas` untuk tabel data.
- `Matplotlib` untuk grafik.
- `scikit-learn` untuk dataset dan model ML.

## Setup Cepat

**Tujuan:** Menyiapkan lingkungan kerja Python yang terisolasi agar library tidak bentrok dengan project lain.
**Kapan dipakai:** Sekali saja di awal, sebelum mulai praktikum.

```bash
python -m venv ml-env
source ml-env/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

**Penjelasan baris per baris:**
- `python -m venv ml-env` — membuat folder virtual environment bernama `ml-env`.
- `source ml-env/bin/activate` — mengaktifkan environment (di Windows: `ml-env\Scripts\activate`).
- `pip install ...` — menginstall semua library yang dibutuhkan selama praktikum.
- `jupyter notebook` — membuka editor notebook di browser. *(Opsional: bisa diganti `jupyter lab` kalau lebih suka tampilan modern.)*

## Contoh Kode Ringkas

**Tujuan:** Memastikan semua library berjalan dan memperkenalkan 4 alat dasar ML: NumPy, Pandas, Matplotlib, dan scikit-learn.
**Kapan dipakai:** Di awal setiap project untuk import library dan cek data.

```python
import numpy as np          # library komputasi angka & array
import pandas as pd          # library tabel data (DataFrame)
import matplotlib.pyplot as plt  # library grafik
from sklearn.datasets import load_iris  # dataset bawaan scikit-learn

# --- NumPy ---
# Membuat array angka lalu menghitung rata-rata.
x = np.array([1, 2, 3, 4, 5])
print("mean:", x.mean())

# --- Pandas ---
# Membuat tabel kecil berisi nama dan nilai, lalu menampilkannya.
df = pd.DataFrame({"nama": ["A", "B", "C"], "nilai": [80, 75, 90]})
print(df)

# --- Visualisasi ---
# Membuat bar chart dari tabel di atas.
df.plot(kind="bar", x="nama", y="nilai", legend=False)
plt.show()  # plt.show() wajib agar grafik muncul

# --- Dataset ML ---
# Memuat dataset Iris (150 bunga, 4 fitur, 3 kelas).
iris = load_iris(as_frame=True)  # as_frame=True agar langsung jadi DataFrame (opsional)
print(iris.frame.head())          # menampilkan 5 baris pertama
```

**Kode opsional:**
- `as_frame=True` pada `load_iris` — jika dihilangkan, data dikembalikan sebagai NumPy array biasa, bukan DataFrame. Ditambahkan supaya lebih mudah dibaca.
- `legend=False` pada `df.plot(...)` — menyembunyikan kotak legenda yang tidak perlu di grafik sederhana.

## Alur Praktikum
1. Jalankan notebook dan pastikan semua library bisa di-import.
2. Coba operasi dasar NumPy (mean, std, reshape).
3. Coba filter data di Pandas.
4. Buat 2 grafik sederhana.
5. Load dataset Iris dan baca 5 baris pertama.

## Tugas
1. Buat array random 20 angka, hitung `mean`, `median`, `std`.
2. Buat DataFrame nilai 10 mahasiswa dan cari 3 nilai tertinggi.
3. Buat 2 grafik dari data nilai.
4. Load dataset `load_wine()` atau `load_breast_cancer()` lalu tulis 3 insight singkat.

## Output Pengumpulan
- 1 file notebook: `NIM_Nama_Pertemuan01.ipynb`
- Semua cell sudah dijalankan dan ada output.

---
Fokus utama: lancar pakai tools dasar dulu. Model ML akan mulai di pertemuan berikutnya.
