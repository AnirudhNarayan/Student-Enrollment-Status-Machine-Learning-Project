# 🎓 Student Dropout Prediction - ML Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy 74.74%](https://img.shields.io/badge/accuracy-74.74%25-brightgreen.svg)](https://github.com/AnirudhNarayan/Student-Enrollment-Status---Machine-Learning-Project)

---

## 🚀 Overview

Welcome to **Student Dropout Prediction** – a comprehensive machine learning solution that demonstrates advanced predictive modeling techniques for educational analytics. This project showcases production-ready ML pipelines, ensemble methods, and robust evaluation frameworks that are essential for real-world applications.

Built with modern Python data science stack and following industry best practices, this repository serves as a testament to professional-grade machine learning development, featuring **74.74% accuracy** through sophisticated ensemble modeling and cross-validation techniques.

---

## 🎯 Key Features

### 🤖 **Advanced ML Pipeline**
- **Multiple Algorithms**: Gradient Boosting, Random Forest, Logistic Regression, Ensemble Methods
- **Cross-Validation**: 5-fold CV with 74.21% (±3.42%) reliability
- **Feature Engineering**: Comprehensive feature importance analysis
- **Model Evaluation**: Precision, Recall, F1-Score, ROC-AUC metrics

### 📊 **Professional Analytics**
- **Exploratory Data Analysis**: Target distribution, correlation analysis
- **Performance Metrics**: Detailed classification reports for each class
- **Visualization**: Model comparison plots and feature importance charts
- **Prediction Distribution**: Real-world prediction insights

### 🛠️ **Production Ready**
- **Clean Code**: Well-documented, modular Python implementation
- **Error Handling**: Robust data validation and preprocessing
- **Scalability**: Efficient algorithms suitable for large datasets
- **Reproducibility**: Deterministic results with proper seeding

---

## 📈 Performance Highlights

| Metric | Value | Status |
|--------|-------|--------|
| **Best Model Accuracy** | 74.74% | 🏆 |
| **Cross-Validation** | 74.21% ± 3.42% | ✅ |
| **Ensemble Performance** | 74.61% | 🎯 |
| **Feature Importance** | v_6 (41.33%) | 🔍 |

### 🎯 **Classification Performance**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Dropout (0)** | 79% | 72% | 76% | 249 |
| **Graduate (1)** | 77% | 91% | 83% | 373 |
| **Enrolled (2)** | 53% | 36% | 43% | 138 |

---

## 🏗️ Project Structure

```
Student_Enrollment_Status/
├── 🐍 final_dropout_prediction.py    # Main ML pipeline
├── 📥 download_data.py               # Data acquisition helper
├── 📋 requirements.txt               # Dependencies
├── 📖 README.md                      # This file
├── 📊 PERFORMANCE_SUMMARY.md         # Detailed metrics
├── 🚫 .gitignore                     # Git ignore rules
├── 📤 submission_final.csv           # Best model predictions
└── 📤 submission_ensemble.csv        # Ensemble predictions
```

---

## 🚀 Quick Start

### 1. **Clone Repository**
```bash
git clone https://github.com/AnirudhNarayan/Student-Enrollment-Status---Machine-Learning-Project.git
cd Student-Enrollment-Status---Machine-Learning-Project
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Get Data**
```bash
# Option 1: Download sample data for testing
python download_data.py

# Option 2: Download from Kaggle
# Visit: https://www.kaggle.com/datasets/akshaydattatraykhare/student-dropout-prediction
```

### 4. **Run Analysis**
```bash
python final_dropout_prediction.py
```

---

## 📊 Dataset Information

| Feature | Description | Type |
|---------|-------------|------|
| **Features** | 41 numerical variables (v_0 to v_40) | Continuous |
| **Target** | Student status (0: Dropout, 1: Graduate, 2: Enrolled) | Categorical |
| **Training Set** | 3,798 samples | Balanced |
| **Test Set** | 1,630 samples | Unseen |

---

## 🔧 Technical Implementation

### **Machine Learning Stack**
- **Scikit-learn**: Core ML algorithms and preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization

### **Key Algorithms**
1. **Gradient Boosting Classifier** - Best performer (74.74%)
2. **Random Forest Classifier** - Robust ensemble method
3. **Logistic Regression** - Interpretable baseline
4. **Voting Classifier** - Ensemble combination

### **Evaluation Framework**
- **Accuracy Score**: Overall performance metric
- **Classification Report**: Per-class detailed metrics
- **Cross-Validation**: Robust model validation
- **Feature Importance**: Model interpretability

---

## 🎯 Model Performance Deep Dive

### **🏆 Best Model: Gradient Boosting**
- **Accuracy**: 74.74%
- **Cross-Validation**: 74.21% (±3.42%)
- **Key Strength**: Excellent performance on Graduate class (91% recall)

### **🔍 Feature Importance Analysis**
| Rank | Feature | Importance | Impact |
|------|---------|------------|--------|
| 1 | v_6 | 41.33% | 🎯 Dominant |
| 2 | v_22 | 11.82% | 📈 High |
| 3 | v_27 | 6.18% | 📊 Medium |
| 4 | v_19 | 5.04% | 📊 Medium |

### **📈 Prediction Distribution**
- **Dropout**: 489 students (30.0%)
- **Graduate**: 956 students (58.7%)
- **Enrolled**: 183 students (11.3%)

---

## 🛠️ Issues Resolved

| Issue | Problem | Solution |
|-------|---------|----------|
| **Data Leakage** | `fit_transform()` on test data | Proper train/test split |
| **Path Issues** | Kaggle-specific paths | Local file handling |
| **Missing Evaluation** | No performance metrics | Comprehensive evaluation |
| **Incomplete Ensemble** | Unused voting classifier | Full ensemble implementation |
| **No Cross-Validation** | Single train/test split | 5-fold cross-validation |



<div align="center">

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ for the ML community*

</div>
