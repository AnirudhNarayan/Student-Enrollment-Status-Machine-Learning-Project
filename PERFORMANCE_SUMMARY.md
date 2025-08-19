# Student Dropout Prediction - Performance Summary

## 🏆 **Best Model Performance**

**Gradient Boosting Classifier** achieved the best performance with:
- **Overall Accuracy**: 74.74%
- **Cross-Validation Accuracy**: 74.21% (±3.42%)

## 📊 **Complete Model Comparison**

| Model | Accuracy | Macro Avg Precision | Macro Avg Recall | Macro Avg F1-Score |
|-------|----------|-------------------|------------------|-------------------|
| **Gradient Boosting** | **74.74%** | **70.0%** | **66.0%** | **67.0%** |
| Ensemble | 74.61% | 69.0% | 65.0% | 66.0% |
| Random Forest | 74.47% | 70.0% | 65.0% | 66.0% |
| Logistic Regression | 72.37% | 65.0% | 62.0% | 62.0% |

## 🎯 **Detailed Classification Reports**

### **Gradient Boosting (Best Model)**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Dropout (0)** | 79% | 72% | 76% | 249 |
| **Graduate (1)** | 77% | 91% | 83% | 373 |
| **Enrolled (2)** | 53% | 36% | 43% | 138 |
| **Overall** | **70%** | **66%** | **67%** | **760** |

### **Ensemble Model**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Dropout (0)** | 80% | 73% | 76% | 249 |
| **Graduate (1)** | 76% | 92% | 83% | 373 |
| **Enrolled (2)** | 53% | 30% | 39% | 138 |
| **Overall** | **69%** | **65%** | **66%** | **760** |

### **Random Forest**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Dropout (0)** | 79% | 72% | 76% | 249 |
| **Graduate (1)** | 75% | 92% | 83% | 373 |
| **Enrolled (2)** | 56% | 30% | 39% | 138 |
| **Overall** | **70%** | **65%** | **66%** | **760** |

### **Logistic Regression**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Dropout (0)** | 76% | 74% | 75% | 249 |
| **Graduate (1)** | 74% | 90% | 81% | 373 |
| **Enrolled (2)** | 46% | 22% | 30% | 138 |
| **Overall** | **65%** | **62%** | **62%** | **760** |

## 🔍 **Cross-Validation Results**

**5-Fold Cross-Validation on Gradient Boosting:**
- **Individual Scores**: [75.99%, 75.45%, 74.79%, 71.17%, 73.64%]
- **Mean Accuracy**: 74.21%
- **Standard Deviation**: ±3.42%

## 📈 **Feature Importance Analysis**

**Top 10 Most Important Features:**

| Rank | Feature | Importance | Percentage |
|------|---------|------------|------------|
| 1 | v_6 | 0.413266 | 41.33% |
| 2 | v_22 | 0.118206 | 11.82% |
| 3 | v_27 | 0.061762 | 6.18% |
| 4 | v_19 | 0.050385 | 5.04% |
| 5 | v_4 | 0.041498 | 4.15% |
| 6 | v_17 | 0.036448 | 3.64% |
| 7 | v_31 | 0.035382 | 3.54% |
| 8 | v_10 | 0.026826 | 2.68% |
| 9 | v_5 | 0.026085 | 2.61% |
| 10 | v_33 | 0.018559 | 1.86% |

## 📊 **Prediction Distributions**

### **Test Set Predictions (1,628 samples)**

**Best Model (Gradient Boosting):**
- Dropout: 489 students (30.0%)
- Graduate: 956 students (58.7%)
- Enrolled: 183 students (11.3%)

**Ensemble Model:**
- Dropout: 485 students (29.8%)
- Graduate: 973 students (59.7%)
- Enrolled: 170 students (10.4%)

## 🎯 **Key Insights**

1. **Best Performance**: Gradient Boosting achieves the highest accuracy (74.74%)
2. **Class Imbalance**: The model performs well on Graduate class (91% recall) but struggles with Enrolled class (36% recall)
3. **Feature Dominance**: Feature v_6 alone contributes 41.33% to the model's decision-making
4. **Robust Validation**: Cross-validation confirms the model's reliability (74.21% ± 3.42%)
5. **Ensemble Performance**: Ensemble model provides consistent performance (74.61%)

## 📁 **Output Files**

- `submission_final.csv` - Best model (Gradient Boosting) predictions
- `submission_ensemble.csv` - Ensemble model predictions

---

*Generated on: $(date)*
*Model: Gradient Boosting Classifier*
*Best Accuracy: 74.74%*
