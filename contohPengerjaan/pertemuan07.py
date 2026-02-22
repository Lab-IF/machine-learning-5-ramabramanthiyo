"""
Pertemuan 7: Clustering (K-Means & Hierarchical)
================================================
Contoh pengerjaan lengkap sesuai materi pertemuan-07.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ============================================================
# Contoh Kode: K-Means & Hierarchical
# ============================================================
print("=== Contoh: K-Means & Hierarchical ===")

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
labels_k = kmeans.fit_predict(X)
print("Inertia:", round(kmeans.inertia_, 2))
print("Silhouette K-Means:", round(silhouette_score(X, labels_k), 3))

# Hierarchical
agg = AgglomerativeClustering(n_clusters=4)
labels_h = agg.fit_predict(X)
print("Silhouette Hierarchical:", round(silhouette_score(X, labels_h), 3))

# ============================================================
# Menentukan K Optimal (Elbow + Silhouette)
# ============================================================
print("\n=== Menentukan K Optimal ===")

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    print(f"  K={k:2d} → Inertia: {km.inertia_:>8.1f}  Silhouette: {silhouette_score(X, labels):.3f}")

best_k = list(K_range)[np.argmax(silhouettes)]
print(f"\n→ K optimal berdasarkan silhouette tertinggi: K={best_k}")

# Elbow plot + Silhouette plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(K_range), inertias, "bo-")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")

axes[1].plot(list(K_range), silhouettes, "ro-")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score")

plt.tight_layout()
plt.savefig("pertemuan07_elbow_silhouette.png")
plt.close()
print("Grafik Elbow + Silhouette disimpan ke pertemuan07_elbow_silhouette.png")

# ============================================================
# TUGAS: Clustering dataset pelanggan (sintetis)
# ============================================================
print("\n=== Tugas: Segmentasi Pelanggan (Sintetis) ===")

np.random.seed(42)
n = 200
income = np.concatenate([
    np.random.normal(30, 5, 50),   # segmen 1: low income, low spending
    np.random.normal(30, 5, 50),   # segmen 2: low income, high spending
    np.random.normal(70, 8, 50),   # segmen 3: high income, low spending
    np.random.normal(70, 8, 50),   # segmen 4: high income, high spending
])
spending = np.concatenate([
    np.random.normal(20, 5, 50),
    np.random.normal(70, 5, 50),
    np.random.normal(20, 5, 50),
    np.random.normal(70, 5, 50),
])
X_cust = np.column_stack([income, spending])

# Tugas 1: Clustering
km_cust = KMeans(n_clusters=4, random_state=42)
labels_cust = km_cust.fit_predict(X_cust)

# Tugas 2: Profil tiap cluster
print("\nProfil Cluster:")
print(f"{'Cluster':>8}  {'Avg Income':>12}  {'Avg Spending':>13}  {'Count':>6}")
for c in sorted(set(labels_cust)):
    mask = labels_cust == c
    avg_inc = income[mask].mean()
    avg_sp = spending[mask].mean()
    count = mask.sum()
    print(f"{c:>8d}  {avg_inc:>12.1f}  {avg_sp:>13.1f}  {count:>6d}")

# Tugas 3: Beri nama segmen
print("\nNama Segmen:")
for c in sorted(set(labels_cust)):
    mask = labels_cust == c
    avg_inc = income[mask].mean()
    avg_sp = spending[mask].mean()
    if avg_inc > 50 and avg_sp > 50:
        name = "Premium"
    elif avg_inc > 50 and avg_sp <= 50:
        name = "Saver"
    elif avg_inc <= 50 and avg_sp > 50:
        name = "Spender"
    else:
        name = "Budget"
    print(f"  Cluster {c}: {name} (income={avg_inc:.0f}, spending={avg_sp:.0f})")

# Visualisasi cluster
plt.figure(figsize=(7, 5))
for c in sorted(set(labels_cust)):
    mask = labels_cust == c
    plt.scatter(income[mask], spending[mask], label=f"Cluster {c}", alpha=0.7)
plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("Segmentasi Pelanggan (K-Means)")
plt.legend()
plt.tight_layout()
plt.savefig("pertemuan07_segmentasi.png")
plt.close()
print("Grafik segmentasi disimpan ke pertemuan07_segmentasi.png")

# Tugas 4: 3 rekomendasi bisnis
print("\n=== 3 Rekomendasi Bisnis ===")
print("1. Segmen 'Premium' (income & spending tinggi): berikan loyalty program")
print("   dan produk eksklusif untuk mempertahankan pengeluaran mereka.")
print("2. Segmen 'Saver' (income tinggi, spending rendah): buat promo menarik")
print("   untuk mendorong mereka belanja lebih banyak.")
print("3. Segmen 'Budget' (income & spending rendah): tawarkan produk terjangkau")
print("   dan diskon agar tetap menjadi pelanggan setia.")

print("\n✅ Pertemuan 07 selesai.")
