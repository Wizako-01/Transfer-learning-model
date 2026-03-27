# Explainable Transfer Learning for Multi-Class Chest X-ray Classification

## Overview
This project implements a transfer learning–based deep learning framework for the classification of chest X-ray (CXR) images into four categories: **Normal, Pneumonia, COVID-19, and Tuberculosis (TB)**.

The study evaluates and compares the performance of three pretrained convolutional neural network architectures:
- DenseNet-121  
- ResNet-50  
- EfficientNet-B0  

The objective is to develop a reliable and explainable AI system for pulmonary disease diagnosis using a Nigerian dataset.

## Dataset
- **Source:** Nigerian Chest X-ray Dataset (Aminu Kano Teaching Hospital)  
- **Total images:** 2,600  
- **Classes:** 4 (Normal, Pneumonia, COVID-19, TB)  
- **Split:**
  - Training: 2,000 images  
  - Testing: 600 images  

## Methodology

### Preprocessing
- Image resizing to 224 × 224  
- Normalization using ImageNet statistics  
- Data augmentation:
  - Random horizontal flip  
  - Random rotation  

### Model Training
A two-stage transfer learning approach was used:

1. **Feature Extraction**
   - Pretrained backbone frozen  
   - Only classification head trained  

2. **Fine-Tuning**
   - Entire network unfrozen  
   - Reduced learning rate  

### Model Architecture
Custom classification head:
- Fully connected layer (256 units)  
- ReLU activation  
- Dropout (0.5)  
- Output layer (4 classes)

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- F1-score  
- Confusion Matrix  
- ROC-AUC (macro-average, One-vs-Rest)

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| ResNet-50 | 92.67% | 0.9971 |
| DenseNet-121 | 91.67% | 0.9940 |
| EfficientNet-B0 | 90.67% | 0.9941 |

DenseNet-121 demonstrated the most balanced performance across classes and produced the most clinically relevant feature visualizations.

## Explainability
Grad-CAM was applied to generate class activation maps, highlighting regions of the lung contributing to model predictions. DenseNet-121 showed the most anatomically meaningful localization.

## Tech Stack
- Python  
- PyTorch  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Google Colab (GPU)

## Project Structure
```text
Transfer-learning-model/
│
├── TL_work.ipynb
└── README.md
