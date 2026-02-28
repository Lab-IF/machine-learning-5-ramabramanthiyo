# Pertemuan 3: Linear Regression & Polynomial Regression

## Tujuan
Peserta mampu:
- Membangun model regresi linear sederhana dan multiple.
- Menggunakan polynomial regression untuk pola non-linear.
- Mengevaluasi model dengan `MAE`, `RMSE`, dan `R²`.

## Konsep Inti
- Regresi dipakai untuk memprediksi nilai angka (kontinu).
- Linear regression: hubungan garis lurus.
- Polynomial regression: menambah pangkat fitur agar pola melengkung bisa dipelajari.

## Contoh 1: Simple Linear Regression

**Tujuan:** Memprediksi satu nilai angka (`y`) berdasarkan satu fitur (`X`) menggunakan garis lurus. Ini adalah bentuk regresi paling sederhana.
**Kapan dipakai:** Saat hubungan antara fitur dan target terlihat mendekati garis lurus (misal: luas rumah vs harga).

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

# Membuat data sintetis: y = 3x + 2 + noise
np.random.seed(42)                       # agar hasil bisa direproduksi (opsional tapi disarankan)
X = np.random.rand(100, 1) * 10         # 100 titik data, fitur antara 0–10
y = 3*X[:, 0] + 2 + np.random.randn(100) # target = 3x + 2, ditambah noise acak

# Membagi data: 80% latih, 20% uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model lalu memprediksi data uji
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi: R² mendekati 1 = bagus, RMSE mendekati 0 = bagus
print("R2:", round(r2_score(y_test, y_pred), 3))
print("RMSE:", round(root_mean_squared_error(y_test, y_pred), 3))
```

**Kode opsional:**
- `np.random.seed(42)` — mengunci angka random agar hasil selalu sama tiap kali dijalankan. Boleh dihilangkan, tapi hasil akan berubah-ubah.
- Untuk menghitung MSE (tanpa akar), gunakan `mean_squared_error(y_test, y_pred)` sebagai gantinya.

## Contoh 2: Multiple Linear Regression

**Tujuan:** Memprediksi target menggunakan **banyak fitur** sekaligus (misal: prediksi harga rumah dari pendapatan, lokasi, jumlah kamar, dll).
**Kapan dipakai:** Saat ada lebih dari satu variabel yang mempengaruhi target.

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Dataset California Housing: ~20 ribu rumah, 8 fitur, target = median harga rumah.
data = fetch_california_housing(as_frame=True)
X = data.frame.drop(columns=["MedHouseVal"])  # semua kolom kecuali target
y = data.frame["MedHouseVal"]                  # kolom target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# model.score() mengembalikan R² secara langsung
print("R2 test:", round(model.score(X_test, y_test), 3))
```

**Kode opsional:**
- `as_frame=True` — mengembalikan data sebagai DataFrame Pandas agar nama kolom mudah dibaca. Tanpa flag ini, data berupa NumPy array.
- `model.score(X_test, y_test)` — shortcut untuk `r2_score(y_test, model.predict(X_test))`.

## Contoh 3: Polynomial Regression

**Tujuan:** Menangkap hubungan **non-linear** (melengkung) antara fitur dan target. Fitur asli `x` ditambah versi pangkatnya (`x²`, `x³`, dst) lalu tetap dilatih dengan Linear Regression.
**Kapan dipakai:** Saat scatter plot menunjukkan pola melengkung, bukan garis lurus.

```python
from sklearn.preprocessing import PolynomialFeatures

# degree=2 berarti: tambahkan kolom x² di samping x.
# include_bias=False berarti tidak menambah kolom konstanta (sudah ditangani LinearRegression).
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train[[X.columns[0]]])  # fit + transform data latih
X_test_poly = poly.transform(X_test[[X.columns[0]]])          # transform saja data uji

poly_model = LinearRegression().fit(X_train_poly, y_train)
print("R2 poly:", round(poly_model.score(X_test_poly, y_test), 3))
```

**Kode opsional:**
- `degree=2` — bisa diubah ke 3, 4, dst. Semakin tinggi, semakin fleksibel tapi risiko **overfitting** meningkat.
- `include_bias=False` — boleh dihilangkan; `LinearRegression` sudah menambah intercept sendiri, jadi bias di sini tidak perlu.

## Tugas
1. Buat simple linear regression (dataset bebas) + visualisasi garis regresi.
2. Buat multiple linear regression (minimal 3 fitur).
3. Bandingkan degree 1 vs degree 2 (polynomial) dan jelaskan mana lebih baik.
4. Laporkan `MAE`, `RMSE`, `R²`.

## Output
- Notebook: `NIM_Nama_Pertemuan03.ipynb`
