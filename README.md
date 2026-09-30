# 🤖 AI & ML Neural Network & Classification — Python

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
</p>

<p align="center">
  <strong>DevSphere Internship — Week 4</strong>
</p>

<p align="center">
  Practical implementation of a basic neural network and a classification model using Python and Scikit-learn.
</p>

---

## 📌 Overview

This repository contains the practical work completed for **Week 4 of the DevSphere Internship**.

The week focuses on two fundamental machine learning areas:

- 🤖 **AI Task:** Neural Networks Basics
- 📊 **ML Task:** Classification Models

Both tasks are implemented in a single Google Colab notebook using publicly available datasets provided through Scikit-learn.

The notebook demonstrates a complete machine learning workflow, including dataset inspection, data preparation, training, prediction, evaluation, and visualization.

---

## 📂 Repository Structure

```text
ai_ml_neural_classification_python/
│
├── AI_ML_Week4_Neural_Classification.ipynb
├── README.md
└── DevSphere_Week4_Report.docx
```

---

# 🤖 Task 1 — Neural Network Basics

## 📊 Dataset

**Digits Dataset**

The Scikit-learn Digits dataset contains handwritten digit images represented as numerical pixel features.

### Dataset Information

| Property | Value |
|---|---:|
| Total Samples | 1,797 |
| Features | 64 |
| Image Size | 8 × 8 |
| Classes | 10 |
| Class Labels | 0–9 |
| Training Samples | 1,437 |
| Testing Samples | 360 |

The notebook displays sample images from the dataset before model training to provide a visual understanding of the data.

## 🧠 Neural Network Model

**Multi-Layer Perceptron (MLP) Neural Network**

The implementation uses a Scikit-learn pipeline with feature scaling followed by an MLP classifier.

```text
StandardScaler
      ↓
MLPClassifier
      ↓
Hidden Layer: 64 neurons
      ↓
ReLU Activation
      ↓
Adam Optimizer
```

## 🔬 Workflow

```text
Digits Dataset
      ↓
Dataset Inspection
      ↓
Sample Image Visualization
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
MLP Neural Network
      ↓
Model Training
      ↓
Predictions
      ↓
Accuracy & Classification Report
      ↓
Confusion Matrix + Training Loss
```

## 📈 Results

**Accuracy: 98.06%**

| Class | Precision | Recall | F1 Score |
|---:|---:|---:|---:|
| 0 | 1.0000 | 0.9722 | 0.9859 |
| 1 | 0.9444 | 0.9444 | 0.9444 |
| 2 | 0.9459 | 1.0000 | 0.9722 |
| 3 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.9730 | 1.0000 | 0.9863 |
| 5 | 1.0000 | 0.9730 | 0.9863 |
| 6 | 1.0000 | 1.0000 | 1.0000 |
| 7 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 0.9697 | 0.9143 | 0.9412 |
| 9 | 0.9730 | 1.0000 | 0.9863 |

**Macro Average F1:** 0.9803  
**Weighted Average F1:** 0.9805

### 🔎 Sample Predictions

```text
1. Actual: 5 | Predicted: 5
2. Actual: 2 | Predicted: 2
3. Actual: 8 | Predicted: 8
4. Actual: 1 | Predicted: 8
5. Actual: 7 | Predicted: 7
```

## 📸 Task 1 Results

### 1. Dataset Information & Sample Images

![Task 1 - Dataset and Sample Images](11.PNG)

### 2. Model Performance & Confusion Matrix

![Task 1 - Model Performance](12.PNG)

### 3. Training Loss & Sample Predictions

![Task 1 - Training Loss](13.PNG)

---

# 📊 Task 2 — Classification Model

## 📚 Dataset

**Breast Cancer Wisconsin Dataset**

The dataset contains numerical measurements used to classify samples into two categories:

- Malignant
- Benign

### Dataset Information

| Property | Value |
|---|---:|
| Total Samples | 569 |
| Features | 30 |
| Classes | 2 |
| Training Samples | 455 |
| Testing Samples | 114 |

## 🤖 Classification Model

**Logistic Regression**

The implementation uses a Scikit-learn pipeline with feature scaling followed by Logistic Regression.

```text
StandardScaler
      ↓
Logistic Regression
      ↓
Predictions
```

## 🔬 Workflow

```text
Breast Cancer Dataset
          ↓
Dataset Inspection
          ↓
Class Distribution
          ↓
Train/Test Split
          ↓
Feature Scaling
          ↓
Logistic Regression
          ↓
Model Training
          ↓
Predictions
          ↓
Accuracy
          ↓
Classification Report
          ↓
Confusion Matrix + ROC Curve
```

## 📈 Results

**Accuracy: 98.25%**

### Classification Report

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Malignant | 0.9762 | 0.9762 | 0.9762 | 42 |
| Benign | 0.9861 | 0.9861 | 0.9861 | 72 |
| Accuracy | — | — | 0.9825 | 114 |
| Macro Avg | 0.9812 | 0.9812 | 0.9812 | 114 |
| Weighted Avg | 0.9825 | 0.9825 | 0.9825 | 114 |

### Confusion Matrix

```text
                 Predicted
              Malignant  Benign

Actual
Malignant         41        1
Benign             1       71
```

### ROC Curve

**AUC: 1.00**

### 🔎 Sample Predictions

```text
1. Actual: malignant | Predicted: malignant
2. Actual: benign | Predicted: benign
3. Actual: malignant | Predicted: malignant
4. Actual: benign | Predicted: benign
5. Actual: malignant | Predicted: malignant
```

## 📸 Task 2 Results

### 1. Dataset Information & Class Distribution

![Task 2 - Dataset and Class Distribution](21.PNG)

### 2. Classification Report & Confusion Matrix

![Task 2 - Classification Results](22.PNG)

### 3. ROC Curve & Sample Predictions

![Task 2 - ROC Curve](23.PNG)

---

# 🛠️ Technologies & Tools

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,github,vscode" alt="Technologies">
</p>

| Technology / Tool | Purpose |
|---|---|
| 🐍 Python | Programming and model implementation |
| 🤖 Scikit-learn | Datasets, preprocessing, models, and evaluation |
| 📈 Matplotlib | Graphs and visualizations |
| 📓 Google Colab | Notebook development and execution |
| 🔗 GitHub | Repository hosting and project management |

---

# 📚 Concepts Practiced

## 🤖 Neural Networks

- Basic neural network architecture
- Multi-Layer Perceptron
- Feature scaling
- Model training
- Predictions
- Training loss
- Confusion matrix
- Classification metrics

## 📊 Classification

- Binary classification
- Logistic Regression
- Feature scaling
- Train/test splitting
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC

---

# 📊 Evaluation Summary

| Task | Dataset | Model | Main Result |
|---|---|---|---:|
| 🤖 Neural Network | Digits Dataset | MLP Neural Network | **98.06% Accuracy** |
| 📊 Classification | Breast Cancer Wisconsin Dataset | Logistic Regression | **98.25% Accuracy** |

---

# 🚀 How to Run

The complete implementation is contained in a single Google Colab notebook.

1. Open `AI_ML_Week4_Neural_Classification.ipynb`.
2. Open the notebook using Google Colab.
3. Run the Task 1 code block.
4. Review the neural network outputs and visualizations.
5. Run the Task 2 code block.
6. Review the classification outputs and visualizations.

The datasets are loaded directly through Scikit-learn, so no separate dataset files are required.

---

# 🎯 Learning Outcomes

Through this week's practical work, the following concepts were practiced:

- Understanding basic neural networks
- Working with classification datasets
- Preparing machine learning data
- Using training and testing splits
- Building Scikit-learn pipelines
- Scaling features
- Training an MLP neural network
- Training a Logistic Regression classifier
- Generating predictions
- Evaluating classification performance
- Interpreting confusion matrices
- Understanding training loss
- Interpreting ROC curves and AUC
- Visualizing machine learning results

---

# 🏢 Internship

**DevSphere Internship — Week 4**

Practical work focused on **Artificial Intelligence and Machine Learning**, with emphasis on **Neural Network Basics and Classification Models**.

---

# 👨‍💻 Author

**Abdul Samad**

GitHub: [abdulsamad010](https://github.com/abdulsamad010)
