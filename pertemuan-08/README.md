# Pertemuan 8: UTS Mini Project ML (End-to-End)

## Tujuan UTS
Menguji kemampuan peserta menjalankan pipeline ML lengkap:
1. Memahami data (EDA).
2. Menyiapkan data (preprocessing).
3. Melatih dan membandingkan model.
4. Menyimpulkan hasil secara jelas.

## Pilih 1 Jenis Project
1. **Classification** (contoh: Titanic, penyakit, spam).
2. **Regression** (contoh: prediksi harga).
3. **Clustering** (contoh: segmentasi pelanggan).

## Struktur Wajib Notebook

### 1) Problem & Dataset (15%)
- Jelaskan masalah, tujuan, dan sumber data.
- Tampilkan ukuran data, nama fitur, dan target.

### 2) EDA (20%)
- Cek missing value.
- Minimal 5 visualisasi.
- Tulis minimal 5 insight.

### 3) Preprocessing (20%)
- Missing value, encoding (jika ada), scaling, split train-test.
- Jelaskan alasan setiap keputusan preprocessing.

### 4) Modeling (30%)
- **Classification/Regression:** minimal 3 model.
- **Clustering:** K-Means + Hierarchical.
- Tampilkan tabel perbandingan metrik.

### 5) Tuning (10%)
- Pilih 1 model terbaik lalu lakukan tuning sederhana (`GridSearchCV` atau setara).

### 6) Kesimpulan (15%)
- Model terbaik dan alasannya.
- Rekomendasi dan kemungkinan perbaikan berikutnya.

## Template Kode Ringkas

**Tujuan:** Template awal yang selalu dipakai di setiap project ML — memuat data, memisahkan fitur dan target, lalu membagi menjadi data latih dan data uji.
**Kapan dipakai:** Selalu, sebagai langkah pertama sebelum preprocessing dan modeling.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Muat dataset dari file CSV
df = pd.read_csv("dataset.csv")

# 2) Pisahkan fitur (X) dan target (y)
X = df.drop(columns=["target"])  # ganti "target" dengan nama kolom target Anda
y = df["target"]

# 3) Bagi data: 80% untuk latih, 20% untuk uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Kode opsional:**
- `test_size=0.2` — bisa diubah ke `0.3` (70-30) tergantung ukuran dataset. Dataset kecil sebaiknya pakai split yang lebih besar untuk training.
- `random_state=42` — mengunci pembagian data agar hasil reproducible. Bisa diganti angka lain.
- Jika dataset bukan CSV, ganti `pd.read_csv(...)` dengan `pd.read_excel(...)`, `pd.read_json(...)`, atau `sns.load_dataset(...)` untuk dataset bawaan seaborn.

## Deliverables
1. `NIM_Nama_UTS_MLPracticum.ipynb`
2. `NIM_Nama_UTS_MLPracticum.pdf`
3. Dataset atau link dataset

## Aturan Singkat
- Notebook harus bisa dijalankan dari atas sampai bawah tanpa error.
- Dilarang plagiarisme.
- Penilaian menekankan **proses + analisis**, bukan hanya angka akurasi.

---
Tips: pilih dataset yang tidak terlalu kecil, dokumentasi singkat tapi jelas, dan fokus pada alur berpikir Anda.
