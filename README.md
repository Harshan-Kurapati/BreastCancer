

# Breast Cancer Diagnosis using Logistic Regression

This project implements a binary classification model using **logistic regression** to predict whether a tumor is malignant or benign based on various features of the tumor cells. The dataset is analyzed, cleaned, and preprocessed before implementing logistic regression from scratch. Additionally, key performance metrics such as accuracy, precision, recall, and F1-score are calculated to evaluate the model's effectiveness.

## Project Overview

This project demonstrates:
1. **Data Cleaning**: Handling missing and irrelevant values in the dataset.
2. **Feature Scaling**: Standardizing features for better model performance.
3. **Logistic Regression**:
   - Implementing logistic regression from scratch using gradient descent and cross-entropy loss.
4. **Evaluation**:
   - Using key performance metrics to assess model accuracy and predictive capability.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Performance Metrics](#performance-metrics)
- [Conclusion](#conclusion)
- [License](#license)

## Installation

To run this project, ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset

The **Breast Cancer Wisconsin (Diagnostic) dataset** contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. Key features include:
- **Features**:
  - `radius_mean`, `texture_mean`, `perimeter_mean`, etc.
  - `compactness_mean`, `concavity_mean`, etc.
  - Additional worst and error statistics.
- **Target Variable**:
  - `diagnosis`: Encoded as `1` for malignant and `0` for benign.

## Project Workflow

### Step 1: Data Cleaning
- Removed unnecessary columns (`id`, `Unnamed: 32`).
- Mapped the `diagnosis` column to numerical values (`M -> 1`, `B -> 0`).
- Checked and confirmed no missing values in the dataset.

### Step 2: Feature Scaling
- Standardized features using **StandardScaler** for optimal model performance.

### Step 3: Logistic Regression from Scratch
- Implemented binary logistic regression using:
  - Hypothesis function: Sigmoid function.
  - Binary cross-entropy loss for cost calculation.
  - Gradient descent optimization to update weights (`theta`).
- Visualized the cost function over iterations to confirm convergence.

### Step 4: Model Evaluation
- Evaluated the model using:
  - **Accuracy**: Proportion of correctly classified instances.
  - **Precision**: Proportion of correctly predicted malignant cases out of all predicted malignant cases.
  - **Recall**: Proportion of correctly predicted malignant cases out of all actual malignant cases.
  - **F1-Score**: Harmonic mean of precision and recall for balanced evaluation.

## Performance Metrics

### Training Dataset
- **Accuracy**: 98.24%
- **F1-Score**: 97.60%
- **Precision**: 96.45%
- **Recall**: 98.79%

### Testing Dataset
- **Accuracy**: 98.25%
- **F1-Score**: 97.62%
- **Precision**: 95.35%
- **Recall**: 100.0%

### Sample Code Snippets

- **Training Logistic Regression**:
```python
def Train_Logistic_Regression(X, y):
    X["bias"] = 1
    theta = np.zeros(X.shape[1])
    iterations = 1000
    alpha = 0.01
    for i in range(iterations):
        z = np.dot(X, theta)
        h_x = 1 / (1 + np.exp(-z))
        cost = (-1 / len(X)) * np.sum(y * np.log(h_x) + (1 - y) * np.log(1 - h_x))
        grad_cost = (1 / len(X)) * np.dot(X.T, (h_x - y))
        theta -= alpha * grad_cost
    return theta
```

- **Evaluating Model Performance**:
```python
def calculate_Metrics(y_pred, y_true):
    print(f"Accuracy: {accuracy_score(y_pred, y_true)}")
    print(f"F1-Score: {f1_score(y_pred, y_true)}")
    print(f"Precision: {precision_score(y_pred, y_true)}")
    print(f"Recall: {recall_score(y_pred, y_true)}")
```

## Conclusion

This project demonstrates:
- The implementation of logistic regression from scratch using Python.
- The importance of data preprocessing, such as feature scaling, for effective model performance.
- High accuracy and precision in diagnosing breast cancer using the given dataset.

### Future Improvements
- Explore other classification models like SVM, decision trees, or ensemble methods.
- Perform hyperparameter tuning for better optimization.

## License

This project is licensed under the MIT License.

