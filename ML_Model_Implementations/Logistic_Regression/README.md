# Logistic Regression From Scratch (NumPy)

This project is for reinforcing the mechanics of logistic regression and L2 regularization (ridge), not for building a production model.

## Overview

- Dataset: Breast Cancer Wisconsin
- Task: Binary classification
- Goal: Implement logistic regression (with L2 regularization) from scratch in NumPy and compare against sklearn-style behavior.

## Results (Test Set)

NumPy implementation
- ROC AUC: 0.997
- Recall (malignant): 1.00
- Precision (malignant): 0.85
- False Negatives: 0
- False Positives: 6

While these results are good, they are evaluated on a small test set. Recall is perfect on this test split, meaning no malignant tumors are missed. Precision is lower at 0.85, 6 benign cases were incorrectly flagged as malignant. This is intentional, this model prioritizes as little false negatives as possible at the cost of a small number of false positives which is acceptable.

Sklearn implementation
- ROC AUC: 0.966
- Recall (malignant): 0.97
- Precision (malignant): 0.94
- False Negatives: 1
- False Positives: 2

Same threshold, precision is a bit higher, but there is a false negative which is a problem.