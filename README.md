# CatBoost Baseline: Klasifikasi Kategori Biaya Kuliah Mahasiswa

Proyek machine learning end-to-end yang menggunakan **CatBoost multiclass + Class Weighting** untuk memprediksi kategori biaya kuliah mahasiswa pada dataset latihan sintetis.

> **Dataset disclaimer:** data dalam repositori ini merupakan dataset sintetis untuk latihan. Data ini bukan dataset resmi GEMASTIK dan bukan data mahasiswa nyata. Hasil evaluasi tidak boleh ditafsirkan sebagai hasil pada populasi mahasiswa sesungguhnya.

## Ringkasan proyek

| Item | Detail |
|---|---|
| Tipe masalah | Klasifikasi multikelas |
| Target | `kategori_biaya_kuliah` |
| Algoritma | CatBoostClassifier |
| Penanganan imbalance | `auto_class_weights="Balanced"` |
| Input | 18 fitur (ID dikeluarkan dari fitur prediktor) |
| Split | Stratified 80:20 |
| Validasi | Stratified 5-Fold Cross-Validation pada data training |
| Pemilihan hyperparameter | Tidak dilakukan; proyek ini sengaja berisi baseline saja |
| Metrik | Accuracy, Balanced Accuracy, Macro F1, Weighted F1, recall/F1 per kelas |

## Hasil referensi dari eksperimen baseline

Angka di bawah berasal dari eksekusi baseline semua fitur yang telah dicatat selama pengembangan proyek. Menjalankan ulang notebook di lingkungan berbeda dapat menghasilkan variasi waktu dan perbedaan kecil antarversi library.

### Cross-validation (training set)

| Metrik | Rata-rata | Standar deviasi |
|---|---:|---:|
| Accuracy | 0.6667 | 0.0433 |
| Balanced Accuracy | 0.6210 | 0.0584 |
| Macro F1 | 0.6148 | 0.0467 |
| Weighted F1 | 0.6659 | 0.0468 |

### Holdout test set

| Metrik | Skor |
|---|---:|
| Accuracy | 0.7000 |
| Balanced Accuracy | 0.6556 |
| Macro F1 | 0.6538 |
| Weighted F1 | 0.7023 |

Recall kelas 1, 2, dan 3 pada eksekusi referensi masing-masing adalah **0.7500**, **0.4545**, dan **0.6098**. Lihat confusion matrix serta laporan per kelas yang dibuat notebook untuk memahami kesalahan prediksi. Hasil test merupakan **benchmark pengembangan**, karena dataset telah dieksplorasi sebelumnya; bukan validasi eksternal yang independen.

## Metodologi

1. Audit data: ukuran dataset, nilai kosong, duplikasi, dan distribusi target.
2. Pembersihan: hapus duplikat penuh, validasi target, dan rapikan tipe fitur.
3. Exclude `id_mahasiswa` dari prediktor untuk menghindari model mempelajari ID.
4. Pertahankan fitur kategorikal sebagai string dan serahkan pemrosesan kategori secara native ke CatBoost. Missing value kategori diberi token `__MISSING__`; missing value numerik dipertahankan sebagai `NaN` agar ditangani CatBoost.
5. Split stratified: 80% training, 20% test.
6. Jalankan 5-Fold CV di data training menggunakan loop manual dengan progress bar.
7. Fit model baseline pada seluruh data training dan evaluasi sekali pada test set.
8. Simpan model, metrik, confusion matrix, feature importance, dan prediksi di folder `artifacts/`.

### Parameter baseline

```python
{
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "auto_class_weights": "Balanced",
    "random_seed": 2026,
    "verbose": 0,
    "allow_writing_files": False,
    "thread_count": 1,
}
```

Model tidak dituning di proyek ini. Menyediakan baseline tetap membuatnya lebih mudah digunakan sebagai pembanding untuk eksperimen berikutnya.

## Struktur repositori

```text
gemastik-catboost-baseline-portfolio/
├── data/
│   └── dataset_latihan_gemastik_mahasiswa.csv
├── notebooks/
│   └── catboost_baseline_end_to_end.ipynb
├── artifacts/                 # hasil lokal setelah notebook dijalankan
├── README.md
├── README_EN.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Menjalankan proyek

Disarankan memakai Python 3.10 atau lebih baru.

```bash
git clone <URL_REPOSITORY_ANDA>
cd gemastik-catboost-baseline-portfolio
python -m venv .venv
```

Aktifkan environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Instal library dan buka Jupyter:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Buka `notebooks/catboost_baseline_end_to_end.ipynb`, lalu jalankan **Kernel → Restart Kernel and Run All Cells**. Notebook mencari file CSV dari root repo atau folder `data/`. Model dan laporan keluaran akan disimpan di `artifacts/`.

## Hasil keluaran

Setelah dijalankan, folder `artifacts/` berisi antara lain:

- `catboost_baseline_class_weighting.cbm`
- `cross_validation_per_fold.csv` dan `cross_validation_summary.csv`
- `test_metrics.csv` dan `classification_report_test.csv`
- `per_class_metrics_test.csv` dan `confusion_matrix_test.csv`
- `feature_importance.csv`
- `test_predictions.csv` dan `metadata.json`

## Batasan dan pekerjaan lanjutan

- Dataset sintetis mungkin tidak merepresentasikan pola data biaya kuliah nyata.
- Kelas target tidak seimbang; evaluasi perlu membaca Macro F1, Balanced Accuracy, serta recall per kelas, bukan Accuracy saja.
- Data test sudah dipakai dalam proses pengembangan terdahulu, sehingga skor referensi harus dianggap hasil pengembangan. Untuk pengujian independen, gunakan data baru yang tidak dipakai saat memilih model.
- Eksperimen lanjutan dapat membandingkan bobot kelas yang ditentukan manual, strategi imbalance lain, atau model pembanding, tetapi semua itu berada di luar notebook baseline ini.

## Lisensi

Kode tersedia berdasarkan lisensi MIT. Dataset sintetis disertakan untuk mereproduksi latihan proyek ini.
