# Red Wine Quality Analysis

Multivariate statistical analysis and predictive modelling on physicochemical properties of red wine.

---

## Overview

Built regression and classification models to predict wine quality scores from lab measurements. Surfaced the key features driving quality using correlation analysis, PCA, and feature importance.

**Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

---

## Dataset

[UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) — `winequality-red.csv`

- 1,599 red wine samples (1,359 after removing duplicates)
- 11 physicochemical input features
- Target: sensory quality score (3–8), rated by human tasters

---

## Project Structure

```
red_wine_analysis/
├── winequality-red.csv
├── RedWineQuality_Final.ipynb
└── README.md
```

---

## What's in the Notebook

| Section | Description |
|---------|-------------|
| 1. Imports | NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn |
| 2. Load & Clean | Load CSV, check missing values, remove 240 duplicates |
| 3. EDA | Quality distribution, correlation heatmap, key bivariate plots |
| 4. Outlier Detection | IQR method per feature |
| 5. PCA | Scree plot, cumulative variance, 2D projection |
| 6. Prepare Data | Train/test split (80/20), StandardScaler |
| 7. Train Models | Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR |
| 8. Model Comparison | RMSE, MAE, R2, 5-fold CV |
| 9. Diagnostics | Predicted vs actual, residual plot |
| 10. Feature Importance | RF tree importance + permutation importance |
| 11. Classification | Random Forest: Low (<=5) vs High (>=6) quality |
| 12. Conclusions | Key findings and limitations |

---

## Key Results

**Top predictors of quality**

| Feature | Correlation with Quality |
|---------|--------------------------|
| alcohol | +0.48 |
| volatile acidity | -0.40 |
| sulphates | +0.25 |
| residual sugar | -0.01 (negligible) |

**Regression performance**

| Model | R2 | RMSE |
|-------|----|------|
| Random Forest | 0.42 | 0.68 |
| Gradient Boosting | 0.38 | 0.71 |
| Linear / Ridge / SVR | ~0.38 | ~0.70 |

**Binary classification** (Low <=5 vs High >=6): **75.7% accuracy**

**PCA:** 9 components explain 95% of the variance across 11 features.

---

## How to Run

1. Place `winequality-red.csv` in the same folder as the notebook
2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Launch the notebook:

```bash
jupyter notebook RedWineQuality_Final.ipynb
```

Run cells top to bottom.

---

## Limitations

- Dataset is imbalanced — ~90% of samples score 5 or 6, making predictions at extreme scores (3, 8) unreliable
- Human taster scores carry subjective noise; this sets a ceiling on how high R2 can realistically go
- No hyperparameter tuning applied — models use reasonable defaults

---

## References

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
*Modeling wine preferences by data mining from physicochemical properties.*
Decision Support Systems, Elsevier, 47(4):547-553, 2009.
