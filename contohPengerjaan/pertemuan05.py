"""
Pertemuan 5: Support Vector Machine (SVM)
=========================================
Contoh pengerjaan lengkap sesuai materi pertemuan-05.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# Load & split dataset
# ============================================================
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Scaling (wajib untuk SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# Tugas 1: Bandingkan kernel linear vs RBF
# ============================================================
models = {
    "SVM Linear": SVC(kernel="linear", C=1),
    "SVM RBF": SVC(kernel="rbf", C=1, gamma="scale"),
}

print("=" * 55)
print("Tugas 1: Perbandingan Kernel Linear vs RBF")
print("=" * 55)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, pred), 3)
    results[name] = acc
    print(f"\n{name} accuracy: {acc}")
    print(classification_report(y_test, pred, target_names=data.target_names))

# ============================================================
# Tugas 2: Tuning sederhana (C, gamma)
# ============================================================
print("=" * 55)
print("Tugas 2: Tuning C dan gamma")
print("=" * 55)

best_acc = 0
best_params = {}

for c_val in [0.1, 1, 10]:
    for gamma_val in [0.01, 0.1, 1]:
        model = SVC(kernel="rbf", C=c_val, gamma=gamma_val)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  C={c_val:>4}, gamma={gamma_val:>4} → accuracy: {acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_params = {"C": c_val, "gamma": gamma_val}

print(f"\n→ Parameter terbaik: C={best_params['C']}, gamma={best_params['gamma']} "
      f"(accuracy: {best_acc:.3f})")

# ============================================================
# Tugas 3: 3 poin kesimpulan
# ============================================================
print("\n" + "=" * 55)
print("Tugas 3: Kesimpulan")
print("=" * 55)
print("1. SVM RBF umumnya lebih baik dari SVM Linear pada dataset Breast Cancer")
print("   karena batas keputusan non-linear lebih cocok untuk data berdimensi tinggi.")
print("2. Parameter C dan gamma sangat berpengaruh pada performa SVM.")
print(f"   Kombinasi terbaik: C={best_params['C']}, gamma={best_params['gamma']}.")
print("3. Scaling data (StandardScaler) wajib dilakukan sebelum SVM,")
print("   karena SVM sangat sensitif terhadap perbedaan skala antar fitur.")

print("\n✅ Pertemuan 05 selesai.")
