# Diabetes-Prediction-DSV-Project-
# Diabetes Prediction using Deep Neural Networks

A comprehensive machine learning project that uses deep neural networks to predict diabetes based on various health indicators. This model achieves **86.5% accuracy** with optimized feature engineering and advanced training techniques.

## 📊 Project Overview

This project implements a sophisticated deep learning approach to predict diabetes using the Pima Indians Diabetes Database. The model incorporates advanced techniques including:

- Enhanced feature engineering
- Class imbalance handling with SMOTE
- Robust data preprocessing
- Optimized neural network architecture

## 🚀 Features

- **Advanced Feature Engineering**: Creates biological markers and metabolic risk scores
- **Intelligent Missing Value Handling**: Group-based median imputation
- **Class Imbalance Solution**: SMOTE oversampling technique
- **Deep Neural Network**: 4-layer architecture with regularization
- **Optimized Training**: Early stopping, learning rate reduction, and batch normalization
- **Threshold Optimization**: F1-score based threshold tuning for better predictions

## 📋 Requirements

tensorflow>=2.19.0
imblearn
pandas
numpy
matplotlib
seaborn
scikit-learn

## 💾 Dataset

The project uses the **Pima Indians Diabetes Database** with the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
- Outcome (Target variable)

**Dataset Requirements**: `diabetes.csv` file with the above columns.

##  Model Architecture

```
Input Layer (with Gaussian Noise)
    ↓
Dense (256 units, swish activation) + BatchNorm + Dropout(0.4)
    ↓
Dense (128 units, swish activation) + BatchNorm + Dropout(0.3)
    ↓
Dense (64 units, swish activation) + BatchNorm + Dropout(0.2)
    ↓
Dense (32 units, swish activation) + BatchNorm + Dropout(0.1)
    ↓
Output Layer (sigmoid activation)
```

### Key Model Features:
- **Optimizer**: Adam with learning rate 0.0008 and gradient clipping
- **Regularization**: L1-L2 regularization (0.005 each)
- **Callbacks**: Early stopping (patience=20) and Learning rate reduction
- **Batch Size**: 64
- **Max Epochs**: 200

## 🔬 Feature Engineering

The model creates several derived features:

1. **Glucose_BP_Ratio**: Relationship between glucose and blood pressure
2. **Insulin_Glucose_Ratio**: Metabolic efficiency indicator
3. **Metabolic_Risk**: Combined risk score from glucose, BMI, and pedigree function
4. **Age_Glucose_Interaction**: Age-related glucose patterns
5. **BP_BMI_Interaction**: Blood pressure and BMI relationship

## 📊 Visualizations

The code generates:
- Training vs Validation Accuracy curves
- Training vs Validation Loss curves
- Confusion Matrix
- Classification Report

##  Acknowledgments

- Pima Indians Diabetes Database from the National Institute of Diabetes and Digestive and Kidney Diseases
- TensorFlow and Keras  for the deep learning framework
- Scikit-learn for preprocessing and evaluation tools


