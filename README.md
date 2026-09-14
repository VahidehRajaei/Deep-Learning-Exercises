# Breast Cancer Diagnosis Classification using Deep Artificial Neural Networks (ANN)

An end-to-end Machine Learning and Deep Learning pipeline designed to classify breast tumors as **Benign** or **Malignant** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. 

This project emphasizes **biomedical feature engineering**, addressing **multicollinearity**, and preventing **data leakage**, achieving high recall and discrimination metrics critical for clinical decision support.

---

## Key Highlights & Performance

- **Test Accuracy:** 99.12% (113/114 correct predictions)
- **Test AUC-ROC Score:** 0.9980
- **Malignant Recall (Sensitivity):** 97.6% (Only 1 False Negative out of 42 cases)
- **Malignant Precision:** 100% (Zero False Positives)

---

## Pipeline & Methodology

### 1. Exploratory Data Analysis (EDA)
- **Class Balance Analysis:** Analyzed the 62.7% (Benign) to 37.3% (Malignant) distribution, guiding stratified train-test splitting.
- **Outlier Detection:** Used class-wise boxplots to show that extreme values in size metrics (`area_mean`, `area_worst`, `area_se`) are genuine biological indicators of advanced malignancies and should not be trimmed.
- **Multicollinearity Discovery:** Generated lower-triangle Pearson correlation heatmaps, identifying near-perfect collinearity ($r \ge 0.98$) between dimension features (`radius`, `perimeter`, and `area`).

### 2. Feature Engineering & Preprocessing
- **Geometric Transformations:** Constructed domain-specific ratios and differences (e.g., area-to-perimeter ratio, worst-to-mean variations) to capture boundary irregularities rather than redundant absolute dimensions.
- **Redundancy Reduction:** Dropped collinear raw features to stabilize gradient updates and accelerate training.
- **Robust Feature Scaling:** Implemented `StandardScaler` to preserve Gaussian distributions and protect normal variances from being crushed by biological extreme values (unlike `MinMaxScaler`).

### 3. Neural Network Architecture
The model is constructed with TensorFlow/Keras using a fully connected Sequential ANN:
- **Input Layer:** Matches processed feature space dimensions.
- **Hidden Layers:** Dense layers with **HeNormal** weight initialization and **ReLU** activation.
- **Regularization:** Integrated **L2 kernel regularization** and **Dropout** layers to prevent overfitting on the small clinical cohort.
- **Output Layer:** Single unit with **Sigmoid** activation for calibrated probability estimation.
- **Optimization:** Trained using the **Adam** optimizer and **Binary Crossentropy** loss.

---

## Evaluation Results

### Confusion Matrix (Test Set, N = 114)

| | Predicted Benign | Predicted Malignant |
| :--- | :---: | :---: |
| **Actual Benign (72)** | **72** (TN) | 0 (FP) |
| **Actual Malignant (42)** | 1 (FN) | **41** (TP) |

### Detailed Classification Report

```text
              precision    recall  f1-score   support

  Benign (0)       0.99      1.00      0.99        72
Malignant (1)      1.00      0.98      0.99        42

    accuracy                           0.99       114
   macro avg       0.99      0.99      0.99       114
weighted avg       0.99      0.99      0.99       114
