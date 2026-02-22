# Pertemuan 5: Support Vector Machine (SVM)

## Tujuan
Peserta mampu:
- Memahami ide dasar SVM (hyperplane, margin, support vector).
- Mencoba kernel `linear` dan `rbf`.
- Memahami efek parameter `C` dan `gamma`.

## Konsep Inti
- SVM mencari batas pemisah terbaik antar kelas.
- `C` mengatur toleransi error.
- `gamma` (untuk RBF) mengatur kompleksitas batas keputusan.

## Contoh Kode Ringkas

**Tujuan:** Membandingkan dua varian SVM (kernel linear vs RBF) untuk klasifikasi tumor payudara (malignant/benign).
**Kapan dipakai:** Saat data sudah bersih dan ada masalah klasifikasi; SVM sangat kuat untuk data berdimensi tinggi.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset Breast Cancer: 569 sample, 30 fitur, 2 kelas.
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# SVM wajib pakai scaling karena sensitif terhadap perbedaan skala fitur.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit di data latih
X_test = scaler.transform(X_test)        # transform saja di data uji (jangan fit lagi!)

# Dua model SVM dengan kernel berbeda.
models = {
    "SVM Linear": SVC(kernel="linear", C=1),
    "SVM RBF": SVC(kernel="rbf", C=1, gamma="scale")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n{name} accuracy:", round(accuracy_score(y_test, pred), 3))
    print(classification_report(y_test, pred, target_names=data.target_names))
```

**Kode opsional:**
- `C=1` — parameter regularisasi. Coba ubah ke `0.1` (lebih toleran) atau `10` (lebih ketat) untuk melihat efeknya.
- `gamma="scale"` — menghitung gamma otomatis berdasarkan jumlah fitur (`1 / (n_features * X.var())`). Bisa diganti angka manual seperti `0.01` atau `0.1`.
- `kernel="rbf"` — bisa diganti `"poly"` (polynomial) atau `"sigmoid"` untuk eksperimen.
- `target_names=data.target_names` — opsional, menampilkan nama kelas agar laporan lebih mudah dibaca.

## Eksperimen Wajib
1. Ubah `C`: `0.1`, `1`, `10`.
2. Ubah `gamma`: `0.01`, `0.1`, `1` (untuk kernel RBF).
3. Catat perubahan accuracy dan jelaskan.

## Tugas
1. Bandingkan kernel linear vs RBF pada dataset yang sama.
2. Lakukan tuning sederhana (`C`, `gamma`) dan pilih model terbaik.
3. Tulis 3 poin kesimpulan.

## Output
- Notebook: `NIM_Nama_Pertemuan05.ipynb`
