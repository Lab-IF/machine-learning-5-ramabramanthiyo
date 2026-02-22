# Pertemuan 6: Ensemble Methods (Random Forest & Gradient Boosting)

## Tujuan
Peserta mampu:
- Memahami ensemble: gabungan banyak model untuk hasil lebih stabil.
- Menggunakan `RandomForest` dan `GradientBoosting`.
- Membaca feature importance dan membandingkan performa model.

## Konsep Inti
- **Bagging** (contoh: Random Forest): banyak model paralel, lalu voting/average.
- **Boosting** (contoh: Gradient Boosting): model dilatih berurutan, fokus pada error sebelumnya.

## Contoh Kode Ringkas (Classification)

**Tujuan:** Membandingkan tiga model klasifikasi — Decision Tree tunggal, Random Forest (bagging), dan Gradient Boosting — untuk melihat seberapa besar peningkatan akurasi dari teknik ensemble.
**Kapan dipakai:** Saat ingin model yang lebih akurat dan stabil daripada satu pohon keputusan saja.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

models = {
    # Baseline: satu pohon keputusan
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    # Bagging: 100 pohon dilatih paralel dengan data acak berbeda
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    # Boosting: pohon dilatih berurutan, memperbaiki error model sebelumnya
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "accuracy:", round(accuracy_score(y_test, pred), 3))
```

**Kode opsional:**
- `n_estimators=100` — jumlah pohon. Coba ubah ke `50` atau `200` untuk melihat pengaruhnya. Lebih banyak biasanya lebih bagus, tapi lebih lambat.
- `random_state=42` — mengunci seed agar hasil bisa direproduksi.

## Contoh Kode Ringkas (Regression)

**Tujuan:** Menunjukkan bahwa ensemble (Random Forest) juga bisa dipakai untuk **regresi** (prediksi angka), bukan hanya klasifikasi.
**Kapan dipakai:** Saat target berupa angka kontinu (misal: kadar gula darah, harga rumah).

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# Dataset Diabetes: 442 pasien, 10 fitur, target = ukuran perkembangan penyakit.
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    db.data, db.target, test_size=0.2, random_state=42
)

# Melatih Random Forest Regressor lalu memprediksi
reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
pred = reg.predict(X_test)

# RMSE = error rata-rata (satuan sama dengan target); R² = seberapa cocok model (0–1).
print("RMSE:", round(root_mean_squared_error(y_test, pred), 3))
print("R2:", round(r2_score(y_test, pred), 3))
```

**Kode opsional:**
- Untuk menghitung MSE (tanpa akar), gunakan `mean_squared_error(y_test, pred)` sebagai gantinya.
- `n_estimators=100` — jumlah pohon. Bisa dinaikkan untuk akurasi lebih baik, tapi waktu komputasi bertambah.

## Tugas
1. Bandingkan Decision Tree vs Random Forest vs Gradient Boosting (classification).
2. Tampilkan 10 fitur terpenting dari Random Forest.
3. Coba ubah `n_estimators` dan lihat pengaruhnya.

## Output
- Notebook: `NIM_Nama_Pertemuan06.ipynb`
