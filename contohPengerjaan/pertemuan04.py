"""
Pertemuan 4: Classification (Logistic Regression & Decision Tree)
================================================================
Contoh pengerjaan lengkap sesuai materi pertemuan-04.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# Load dataset Iris
# ============================================================
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# ============================================================
# Latih 2 model
# ============================================================
lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

# ============================================================
# Tugas 1 & 2: Evaluasi kedua model
# ============================================================
results = {}
for name, model in {"LogReg": lr, "DecisionTree": dt}.items():
    pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, pred), 3)
    results[name] = acc
    print(f"\n{'='*50}")
    print(f"{name} — Accuracy: {acc}")
    print(f"{'='*50}")

    # Tugas 3: Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

    # Tugas 2: Precision, Recall, F1
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=iris.target_names))

# ============================================================
# Tugas 4: Kesimpulan
# ============================================================
print("=" * 50)
print("KESIMPULAN")
print("=" * 50)
best = max(results, key=results.get)
print(f"• LogReg accuracy  : {results['LogReg']}")
print(f"• DecisionTree accuracy: {results['DecisionTree']}")
if results["LogReg"] == results["DecisionTree"]:
    print(f"→ Kedua model memiliki accuracy yang sama ({results[best]}).")
    print("  Logistic Regression lebih sederhana dan interpretatif untuk dataset kecil seperti Iris.")
else:
    print(f"→ {best} memiliki accuracy lebih tinggi ({results[best]}).")
print("  Pada dataset Iris yang kecil dan terpisah jelas, kedua model umumnya perform sangat baik.")

print("\n✅ Pertemuan 04 selesai.")
