# Water Potability Classification — Final Project

**University of Isfahan — Faculty of Computer Engineering**  
* **Course:** Machine Learning Fundamentals  
* **Instructor:** Dr. Maedeh Jamali  
* **Teaching Assistants:** Zahra Mortazavi, Amirtaha Najaf, Miad Kimiagari, Sepehr Rajabi  

---

## 📌 Project Overview

This project implements a complete, end-to-end machine learning pipeline to predict water potability based on 9 chemical and physical features (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity).

### ⚠️ Core Constraint
Every single classification algorithm in this project is implemented **entirely from scratch using pure NumPy**. No pre-built model classes from `scikit-learn`, `xgboost`, or `lightgbm` were used. The use of `scikit-learn` is strictly restricted to:
* `train_test_split` and `StandardScaler` (data preprocessing only)
* `GridSearchCV` (as a generic hyperparameter tuning loop wrapping our custom fit/predict methods)
* `BaseEstimator` / `ClassifierMixin` (for Scikit-Learn pipeline compatibility)
* `sklearn.metrics` (evaluation metrics computation)

---

## 🏗 Pipeline Architecture (Single Notebook)

All three phases of the project are fully integrated into a single Jupyter Notebook (`Water_Potability_Project.ipynb`):

### Phase 1: Preprocessing & Exploratory Data Analysis (EDA)
* **Dataset:** 3,276 samples × 10 columns (9 float features + 1 target).
* **Class Imbalance:** 60.99% non-potable vs. 39.01% potable (handled via stratified splitting and F1-score optimization).
* **Missing Values:** Median imputation on Sulfate (23.84%), pH (14.99%), and Trihalomethanes (4.95%) computed strictly on training data after splitting to prevent data leakage.
* **Visual Insights:** Feature distributions, boxplots for outliers, and correlation heatmaps confirming near-zero linear correlation between features and target—signaling a non-linear problem space.
* **Splitting & Scaling:** 80/20 stratified train/test split followed by `StandardScaler`.

### Phase 2: From-Scratch Supervised Learning (10 Models)
Built on top of a foundational custom CART Decision Tree (`DecisionTreeScratch`), 10 estimators were implemented using vectorized NumPy:

1. **Logistic Regression:** Sigmoid + Batch Gradient Descent with L2 regularization.
2. **KNN:** Vectorized Euclidean distance matrix with uniform/distance-weighted voting.
3. **SVM:** Kernel Pegasos (stochastic sub-gradient descent) with Linear and RBF kernels.
4. **Decision Tree:** CART implementation with Gini and Entropy criteria.
5. **Random Forest:** Manual Bagging with Bootstrap sampling and $\sqrt{d}$ feature subspace splits.
6. **Gradient Boosting:** Sequential regression trees fit on log-loss residuals.
7. **AdaBoost:** Discrete SAMME with decision stumps and sample re-weighting.
8. **MLP Classifier:** Fully-connected feed-forward neural network with manual backpropagation.
9. **XGBoost-Style (Bonus):** Newton boosting ($g$ & $h$) with depth-wise tree growth and regularized gain.
10. **LightGBM-Style (Bonus):** Newton boosting with leaf-wise tree growth.

### Phase 3: Evaluation & Model Benchmarking
Comprehensive benchmarking on the 656-sample test set using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and comparative performance plots.

---

## 📊 Final Results (Sorted by F1-Score)

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **SVM (RBF)** | 0.553 | 0.455 | **0.738** | **0.563** |
| **MLP Classifier** | 0.659 | 0.593 | 0.398 | 0.477 |
| **LightGBM-Style** | 0.663 | 0.607 | 0.387 | 0.473 |
| **Decision Tree** | 0.561 | 0.441 | 0.469 | 0.455 |
| **XGBoost-Style** | 0.666 | 0.633 | 0.344 | 0.446 |
| **Random Forest** | 0.671 | 0.708 | 0.266 | 0.386 |
| **KNN** | 0.590 | 0.458 | 0.277 | 0.345 |
| **Gradient Boosting** | 0.659 | 0.716 | 0.207 | 0.321 |
| **AdaBoost** | 0.608 | 0.492 | 0.125 | 0.199 |
| **Logistic Regression** | 0.610 | 0.000 | 0.000 | 0.000 |

### 🔍 Key Findings
* **SVM RBF Superiority:** SVM with an RBF kernel achieved the highest F1-Score due to its strong Recall, making it effective at detecting potable water samples.
* **Non-Linear Decision Boundary:** Logistic Regression predicted the majority class for all test samples (0 Precision/Recall), confirming that the dataset is non-linearly separable.
* **Ensemble Behavior:** Tree-based ensembles (Random Forest, Gradient Boosting, XGBoost) exhibited high Precision but lower Recall, leaning conservatively toward the majority class.

---

## 📁 Repository Structure & Execution

* `Water_Potability_Project.ipynb`: The primary notebook containing all three phases.
* `Water_Potability_Project_Report_FA.docx`: Detailed project report (Persian).

### How to Run
1. Open `Water_Potability_Project.ipynb` in Kaggle or Jupyter Notebook.
2. Attach the Water Potability dataset.
3. Run all cells sequentially.
> **Note:** Because all algorithms are implemented in pure Python/NumPy without C/Cython optimizations, running the full notebook with GridSearch takes approximately 20–25 minutes.
