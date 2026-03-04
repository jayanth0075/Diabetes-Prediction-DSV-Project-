Diabetes Prediction using Deep Neural Networks
Overview

This project implements a deep learning model to predict diabetes using the Pima Indians Diabetes Dataset. The system combines advanced feature engineering, robust preprocessing, class imbalance handling, and a carefully optimized neural network architecture. The final model achieves 86.5% accuracy with strong F1-score performance through threshold optimization.

Dataset

The model is trained on the Pima Indians Diabetes Database from the National Institute of Diabetes and Digestive and Kidney Diseases.

Features used:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

Outcome (Target Variable)

The dataset file required: diabetes.csv

Project Pipeline
1. Data Preprocessing

Intelligent handling of missing values using group-based median imputation

Feature scaling

Outlier handling

Train-test split

2. Feature Engineering

The model creates additional derived features to improve predictive power:

Glucose_BP_Ratio

Insulin_Glucose_Ratio

Metabolic_Risk

Age_Glucose_Interaction

BP_BMI_Interaction

These engineered features help capture biological relationships and metabolic patterns.

3. Class Imbalance Handling

SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the dataset and improve recall and F1-score.

4. Model Architecture

Input Layer with Gaussian Noise
Dense Layer (256 units, swish activation)
Batch Normalization
Dropout (0.4)

Dense Layer (128 units, swish activation)
Batch Normalization
Dropout (0.3)

Dense Layer (64 units, swish activation)
Batch Normalization
Dropout (0.2)

Dense Layer (32 units, swish activation)
Batch Normalization
Dropout (0.1)

Output Layer (1 unit, sigmoid activation)

5. Training Configuration

Optimizer: Adam (learning rate = 0.0008)

Gradient clipping enabled

L1-L2 regularization (0.005 each)

Batch size: 64

Maximum epochs: 200

Early stopping with patience = 20

Learning rate reduction on plateau

6. Threshold Optimization

Instead of using the default 0.5 threshold, the model selects an optimal classification threshold based on F1-score to improve predictive balance between precision and recall.

Model Performance

Accuracy: 86.5%

Improved recall through SMOTE

Optimized F1-score using threshold tuning

Evaluation using confusion matrix and classification report

Visualizations

The project includes:

Training vs Validation Accuracy curves

Training vs Validation Loss curves

Confusion Matrix

Classification Report

Requirements

tensorflow >= 2.19.0
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn

Acknowledgments

Pima Indians Diabetes Database from the National Institute of Diabetes and Digestive and Kidney Diseases

TensorFlow and Keras for the deep learning framework

Scikit-learn for preprocessing and evaluation tools
