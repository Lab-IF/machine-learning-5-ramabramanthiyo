"""
Pertemuan 3: Linear Regression & Polynomial Regression
======================================================
Contoh pengerjaan lengkap sesuai materi pertemuan-03.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing

# ============================================================
# Contoh 1: Simple Linear Regression
# ============================================================
print("=== Contoh 1: Simple Linear Regression ===")
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X[:, 0] + 2 + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2  :", round(r2_score(y_test, y_pred), 3))
print("RMSE:", round(root_mean_squared_error(y_test, y_pred), 3))

# Visualisasi garis regresi
plt.figure(figsize=(7, 4))
plt.scatter(X_test, y_test, label="Aktual", alpha=0.7)
plt.plot(sorted(X_test[:, 0]), [model.predict([[x]])[0] for x in sorted(X_test[:, 0])],
         color="red", label="Prediksi")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("pertemuan03_simple_lr.png")
plt.close()
print("Grafik disimpan ke pertemuan03_simple_lr.png\n")

# ============================================================
# Contoh 2: Multiple Linear Regression
# ============================================================
print("=== Contoh 2: Multiple Linear Regression ===")
data = fetch_california_housing(as_frame=True)
X = data.frame.drop(columns=["MedHouseVal"])
y = data.frame["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print("R2 test:", round(model.score(X_test, y_test), 3))

# ============================================================
# Contoh 3: Polynomial Regression
# ============================================================
print("\n=== Contoh 3: Polynomial Regression ===")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train[[X.columns[0]]])
X_test_poly = poly.transform(X_test[[X.columns[0]]])

poly_model = LinearRegression().fit(X_train_poly, y_train)
print("R2 poly:", round(poly_model.score(X_test_poly, y_test), 3))

# ============================================================
# TUGAS: contoh pengerjaan
# ============================================================
print("\n=== Tugas 1: Simple LR + Visualisasi ===")
# (Sudah dikerjakan di Contoh 1 dengan visualisasi)
print("Lihat grafik pertemuan03_simple_lr.png")

print("\n=== Tugas 2: Multiple LR (minimal 3 fitur) ===")
print("Menggunakan California Housing (8 fitur).")
y_pred_multi = model.predict(X_test)
print("MAE :", round(mean_absolute_error(y_test, y_pred_multi), 3))
print("RMSE:", round(root_mean_squared_error(y_test, y_pred_multi), 3))
print("R2  :", round(r2_score(y_test, y_pred_multi), 3))

print("\n=== Tugas 3: Bandingkan degree 1 vs 2 ===")
# Degree 1 (linear)
X_train_d1 = X_train[[X.columns[0]]]
X_test_d1 = X_test[[X.columns[0]]]
model_d1 = LinearRegression().fit(X_train_d1, y_train)

# Degree 2 (polynomial) — sudah dihitung di atas
print(f"Degree 1 R2: {round(model_d1.score(X_test_d1, y_test), 3)}")
print(f"Degree 2 R2: {round(poly_model.score(X_test_poly, y_test), 3)}")
print("→ Degree 2 menangkap pola non-linear sehingga R2 lebih tinggi.")

print("\n=== Tugas 4: Metrik Lengkap (MAE, RMSE, R²) ===")
y_pred_poly = poly_model.predict(X_test_poly)
print("MAE :", round(mean_absolute_error(y_test, y_pred_poly), 3))
print("RMSE:", round(root_mean_squared_error(y_test, y_pred_poly), 3))
print("R2  :", round(r2_score(y_test, y_pred_poly), 3))

print("\n✅ Pertemuan 03 selesai.")
