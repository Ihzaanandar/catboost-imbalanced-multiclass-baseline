# CatBoost Baseline for Student Tuition-Fee Category Classification

An end-to-end multiclass classification portfolio project using **CatBoost with class weighting** on a synthetic student tuition-fee dataset.

> **Dataset notice:** The included CSV is synthetic practice data. It is not the official GEMASTIK dataset and contains no real student records. The results must not be interpreted as performance on real student populations.

## Project overview

| Item | Description |
|---|---|
| Problem | Multiclass classification |
| Target | `kategori_biaya_kuliah` |
| Algorithm | CatBoostClassifier |
| Imbalance strategy | `auto_class_weights="Balanced"` |
| Predictors | 18 features, excluding the ID column |
| Split | Stratified 80:20 train/test split |
| Validation | Stratified 5-fold CV on the training set |
| Hyperparameter search | None; this repository intentionally contains only the baseline |
| Metrics | Accuracy, Balanced Accuracy, Macro F1, Weighted F1, per-class recall/F1 |

## Reference results from the baseline experiment

The following values are from the all-feature baseline run recorded during development. Runtime and small numerical differences may vary across environments and package versions.

### Cross-validation (training set)

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.6667 | 0.0433 |
| Balanced Accuracy | 0.6210 | 0.0584 |
| Macro F1 | 0.6148 | 0.0467 |
| Weighted F1 | 0.6659 | 0.0468 |

### Holdout test set

| Metric | Score |
|---|---:|
| Accuracy | 0.7000 |
| Balanced Accuracy | 0.6556 |
| Macro F1 | 0.6538 |
| Weighted F1 | 0.7023 |

Reference recall for classes 1, 2 and 3 was **0.7500**, **0.4545**, and **0.6098**, respectively. Consult the confusion matrix and per-class report for error patterns. The test scores are a **development benchmark**, not independent external validation, because this dataset has previously been explored during development.

## Methodology

1. Audit dataset dimensions, missing values, duplicates, and target distribution.
2. Remove exact duplicate rows, validate the target, and normalize feature data types.
3. Exclude `id_mahasiswa` from predictors to avoid learning an identifier.
4. Keep categorical features as strings for native CatBoost categorical handling. Missing categorical values are mapped to `__MISSING__`; numeric missing values remain as `NaN` for CatBoost to handle.
5. Split data using stratified 80:20 train/test partitioning.
6. Run manual 5-fold CV on training data with a progress bar.
7. Fit one baseline model on the full training split and evaluate on the holdout test set.
8. Export the model, metrics, confusion matrix, feature importance, and predictions to `artifacts/`.

## Baseline parameters

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

No hyperparameter tuning is performed. A fixed baseline is useful as a reproducible point of comparison for later experiments.

## Repository structure

```text
gemastik-catboost-baseline-portfolio/
├── data/dataset_latihan_gemastik_mahasiswa.csv
├── notebooks/catboost_baseline_end_to_end.ipynb
├── artifacts/                 # generated locally after running the notebook
├── README.md
├── README_EN.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Run locally

Python 3.10 or newer is recommended.

```bash
git clone <YOUR_REPOSITORY_URL>
cd gemastik-catboost-baseline-portfolio
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies and launch Jupyter:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/catboost_baseline_end_to_end.ipynb` and choose **Kernel → Restart Kernel and Run All Cells**. The notebook locates the CSV in the repo root or `data/`. Model and reports are written to `artifacts/`.

## Limitations and future work

- Synthetic data may not capture real-world tuition-fee patterns.
- The target is imbalanced; review Macro F1, Balanced Accuracy and class-level recall instead of Accuracy alone.
- The test set has been involved in earlier development decisions, so the reference score should be treated as exploratory. Use a newly collected untouched dataset for independent evaluation.
- Possible follow-up experiments include manually defined class weights, other imbalance strategies, or model comparisons; these are outside the scope of this baseline-only notebook.

## License

Code is licensed under the MIT License. The included synthetic dataset is provided to reproduce this practice project.
