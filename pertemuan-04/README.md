# Pertemuan 4: Classification (Logistic Regression & Decision Tree)

## Tujuan
Peserta mampu:
- Membedakan regresi vs klasifikasi.
- Melatih model `LogisticRegression` dan `DecisionTreeClassifier`.
- Mengevaluasi model dengan confusion matrix, accuracy, precision, recall, F1.

## Konsep Inti
- Klasifikasi memprediksi kelas (contoh: 0/1, spam/tidak).
- Logistic Regression menghasilkan probabilitas kelas.
- Decision Tree membuat aturan if-else berbentuk pohon.

## Contoh Kode Ringkas (Iris)

**Tujuan:** Melatih dua model klasifikasi (Logistic Regression dan Decision Tree) pada dataset Iris, lalu membandingkan performanya.
**Kapan dipakai:** Saat ingin mengklasifikasikan data ke dalam kategori (misal: jenis bunga, spam/bukan, sakit/sehat).

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset Iris: 150 bunga, 4 fitur, 3 kelas (setosa, versicolor, virginica).
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Logistic Regression: model statistik yang menghitung probabilitas tiap kelas.
lr = LogisticRegression(max_iter=200).fit(X_train, y_train)

# Decision Tree: model pohon keputusan yang memecah data berdasarkan aturan if-else.
dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

# Loop untuk mengevaluasi kedua model sekaligus.
for name, model in {"LogReg": lr, "DecisionTree": dt}.items():
    pred = model.predict(X_test)
    print(f"\n{name} accuracy:", round(accuracy_score(y_test, pred), 3))
    print(confusion_matrix(y_test, pred))           # tabel prediksi vs aktual
    print(classification_report(y_test, pred, target_names=iris.target_names))  # precision, recall, F1
```

**Kode opsional:**
- `max_iter=200` di `LogisticRegression` — menambah iterasi agar model pasti konvergen. Default-nya 100; jika muncul warning "ConvergenceWarning", naikkan angka ini.
- `max_depth=3` di `DecisionTreeClassifier` — membatasi kedalaman pohon agar tidak overfitting. Jika dihilangkan, pohon tumbuh penuh dan cenderung menghafal data latih.
- `random_state=42` — mengunci random seed agar hasil reproducible. Bisa diubah ke angka apa saja.
- `target_names=iris.target_names` di `classification_report` — menampilkan nama kelas alih-alih angka 0/1/2. Opsional tapi mempermudah pembacaan.

## Tugas
1. Jalankan 2 model: Logistic Regression dan Decision Tree.
2. Bandingkan metrik: accuracy, precision, recall, F1.
3. Tampilkan confusion matrix untuk keduanya.
4. Simpulkan model mana yang lebih baik dan alasannya.

## Output
- Notebook: `NIM_Nama_Pertemuan04.ipynb`
