# Logistic Regression Baseline for EHRSHOT Celiac Disease Classification

This directory contains a modular scikit-learn implementation for binary classification of celiac disease diagnosis from EHRSHOT patient records.

## Files
- `config.py`: experiment/configuration dataclass
- `model.py`: model construction
- `training.py`: training and inference utilities
- `evaluation.py`: binary classification metrics
- `test_logreg_baseline.py`: sanity check script

## Assumptions
- Features are already encoded as numeric vectors.
- Labels are binary encoded as 0/1.
- Standardization is performed inside the training pipeline to avoid leakage.
