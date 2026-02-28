# Pertemuan 7: Clustering (K-Means & Hierarchical)

## Tujuan
Peserta mampu:
- Memahami clustering sebagai unsupervised learning.
- Melakukan K-Means dan menentukan jumlah cluster optimal.
- Melakukan Hierarchical Clustering dan membaca dendrogram.

## Konsep Inti
- Clustering mengelompokkan data tanpa label.
- K-Means butuh jumlah cluster `K` di awal.
- Hierarchical membangun struktur cluster bertingkat.
- Evaluasi umum: `inertia` dan `silhouette score`.

## Contoh Kode Ringkas

**Tujuan:** Mengelompokkan data tanpa label ke dalam cluster menggunakan dua metode: K-Means dan Hierarchical Clustering, lalu membandingkan hasilnya.
**Kapan dipakai:** Saat tidak ada label/target (unsupervised), misal: segmentasi pelanggan, pengelompokan dokumen.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Membuat data sintetis: 300 titik, 4 kelompok alami.
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# --- K-Means ---
# Membagi data ke 4 cluster berdasarkan jarak ke centroid terdekat.
kmeans = KMeans(n_clusters=4, random_state=42)
labels_k = kmeans.fit_predict(X)  # fit + langsung dapatkan label cluster
print("Inertia:", round(kmeans.inertia_, 2))          # total jarak ke centroid (makin kecil makin baik)
print("Silhouette:", round(silhouette_score(X, labels_k), 3))  # -1 s/d 1 (makin tinggi makin baik)

# --- Hierarchical ---
# Menggabungkan titik-titik terdekat secara bertahap dari bawah ke atas.
agg = AgglomerativeClustering(n_clusters=4)
labels_h = agg.fit_predict(X)
print("Silhouette Hierarchical:", round(silhouette_score(X, labels_h), 3))
```

**Kode opsional:**
- `n_clusters=4` — jumlah cluster. Gunakan Elbow Method atau Silhouette Score untuk menentukan angka terbaik (lihat bagian "Menentukan K Optimal").
- `random_state=42` — seed K-Means agar inisialisasi centroid konsisten. `AgglomerativeClustering` tidak pakai random seed karena deterministik.
- `_ = make_blobs(...)` — variabel kedua (`_`) adalah label asli, diabaikan karena tujuan clustering adalah menemukan pola tanpa label.

## Menentukan K Optimal (Wajib)
1. Coba `K=2` sampai `K=10`.
2. Simpan nilai inertia dan silhouette.
3. Pilih `K` terbaik berdasarkan grafik elbow + silhouette tertinggi yang masuk akal.

## Tugas
1. Lakukan clustering pada dataset pelanggan (bebas atau buat sintetis).
2. Buat profil ringkas tiap cluster (misal: rata-rata income/spending).
3. Beri nama segmen (contoh: "High value", "Budget").
4. Tulis 3 rekomendasi bisnis berdasarkan cluster.

## Output
- Notebook: `NIM_Nama_Pertemuan07.ipynb`
