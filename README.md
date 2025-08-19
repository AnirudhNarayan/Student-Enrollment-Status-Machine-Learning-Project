# Student Dropout Prediction - Complete Analysis

This project provides a comprehensive machine learning solution for predicting student dropout status. The original notebook had several issues that have been completely fixed and enhanced.

## Issues Fixed from Original Notebook

### 1. **Path Issues**
- **Problem**: Used Kaggle-specific paths (`/kaggle/input/`) that don't work locally
- **Fix**: Updated to use local file paths (`train.csv`, `test.csv`)

### 2. **Data Leakage**
- **Problem**: Used `fit_transform()` on test data, causing data leakage
- **Fix**: Proper train/validation split with `fit_transform()` only on training data and `transform()` on validation/test data

### 3. **Missing Evaluation**
- **Problem**: No model performance evaluation or metrics
- **Fix**: Added comprehensive evaluation including accuracy, AUC, classification reports, and confusion matrices

### 4. **Incomplete Ensemble**
- **Problem**: Voting classifier was created but never used for predictions
- **Fix**: Complete ensemble implementation with proper evaluation and predictions

### 5. **No Cross-Validation**
- **Problem**: No model validation or cross-validation
- **Fix**: Added 5-fold cross-validation for robust model evaluation

### 6. **Missing Data Analysis**
- **Problem**: No exploratory data analysis
- **Fix**: Comprehensive EDA including target distribution, feature correlations, and visualizations

### 7. **No Feature Engineering**
- **Problem**: Basic preprocessing only
- **Fix**: Proper feature scaling, feature importance analysis, and correlation analysis

### 8. **Incomplete Code Structure**
- **Problem**: Incomplete cells and missing functionality
- **Fix**: Complete, well-structured code with proper documentation

## Project Structure

```
Student_Enrollment_Status/
├── final_dropout_prediction.py  # Main Python script (complete solution)
├── download_data.py             # Data download helper script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── PERFORMANCE_SUMMARY.md       # Detailed performance metrics
├── .gitignore                   # Git ignore file
├── submission_final.csv         # Best model predictions
└── submission_ensemble.csv      # Ensemble model predictions
```

**Note**: The large CSV data files (`train.csv`, `test.csv`) are not included in the repository due to size constraints. Use `download_data.py` to get sample data or download from Kaggle.

## Features

### 🔍 **Exploratory Data Analysis**
- Target variable distribution analysis
- Feature correlation analysis
- Missing value detection
- Statistical summaries

### 🤖 **Multiple Machine Learning Models**
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine (SVM)
- Ensemble Model (Voting Classifier)

### 📊 **Comprehensive Evaluation**
- Accuracy and AUC metrics
- Classification reports
- Confusion matrices
- Cross-validation
- Feature importance analysis

### 📈 **Visualizations**
- Target distribution pie chart
- Feature correlation plots
- Model performance comparisons
- Feature importance charts

### 📁 **Output Files**
- `submission_final.csv` - Best model predictions
- `submission_ensemble.csv` - Ensemble model predictions
- `model_performance.png` - Performance comparison plots
- `feature_importance.png` - Feature importance visualization

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AnirudhNarayan/Student_Enrollment_Status.git
   cd Student_Enrollment_Status
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get the data files**:
   ```bash
   # Option 1: Download sample data for testing
   python download_data.py
   
   # Option 2: Download from Kaggle
   # Go to: https://www.kaggle.com/datasets/akshaydattatraykhare/student-dropout-prediction
   # Download and extract train.csv and test.csv to this directory
   ```

## Usage

### Run the Complete Analysis
```bash
python final_dropout_prediction.py
```

This will:
- Load and analyze the data
- Train multiple machine learning models
- Perform cross-validation
- Generate performance reports
- Create submission files

## Dataset Information

The dataset contains student enrollment data with the following structure:

- **Features**: 41 numerical features (v_0 to v_40)
- **Target**: `label` column with 3 classes:
  - 0: Dropout
  - 1: Graduate  
  - 2: Enrolled

- **Training set**: 3,798 samples
- **Test set**: 1,630 samples

## Model Performance Results

### 📊 **Overall Performance Summary**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Gradient Boosting** | **74.74%** | 70.0% | 66.0% | 67.0% |
| **Ensemble** | 74.61% | 69.0% | 65.0% | 66.0% |
| **Random Forest** | 74.47% | 70.0% | 65.0% | 66.0% |
| **Logistic Regression** | 72.37% | 65.0% | 62.0% | 62.0% |

### 🎯 **Detailed Classification Reports**

#### **Best Model: Gradient Boosting (74.74% Accuracy)**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Dropout (0)** | 79% | 72% | 76% | 249 |
| **Graduate (1)** | 77% | 91% | 83% | 373 |
| **Enrolled (2)** | 53% | 36% | 43% | 138 |
| **Overall** | **70%** | **66%** | **67%** | **760** |

#### **Ensemble Model (74.61% Accuracy)**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Dropout (0)** | 80% | 73% | 76% | 249 |
| **Graduate (1)** | 76% | 92% | 83% | 373 |
| **Enrolled (2)** | 53% | 30% | 39% | 138 |
| **Overall** | **69%** | **65%** | **66%** | **760** |

### 🔍 **Cross-Validation Results**

- **5-Fold CV Accuracy**: 74.21% (±3.42%)
- **CV Scores**: [75.99%, 75.45%, 74.79%, 71.17%, 73.64%]

### 📈 **Feature Importance (Top 10)**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | v_6 | 41.33% |
| 2 | v_22 | 11.82% |
| 3 | v_27 | 6.18% |
| 4 | v_19 | 5.04% |
| 5 | v_4 | 4.15% |
| 6 | v_17 | 3.64% |
| 7 | v_31 | 3.54% |
| 8 | v_10 | 2.68% |
| 9 | v_5 | 2.61% |
| 10 | v_33 | 1.86% |

### 📊 **Prediction Distribution**

- **Test Set Predictions (Best Model)**: 
  - Dropout: 489 students (30.0%)
  - Graduate: 956 students (58.7%)
  - Enrolled: 183 students (11.3%)

- **Ensemble Predictions**: 
  - Dropout: 485 students (29.8%)
  - Graduate: 973 students (59.7%)
  - Enrolled: 170 students (10.4%)

## Key Improvements Made

1. **Proper Data Handling**: Fixed all path and data loading issues
2. **Robust Evaluation**: Added comprehensive model evaluation metrics
3. **Feature Analysis**: Included feature importance and correlation analysis
4. **Visualization**: Added informative plots and charts
5. **Cross-Validation**: Ensured model reliability with CV
6. **Ensemble Methods**: Implemented voting classifier for better performance
7. **Documentation**: Added comprehensive comments and documentation

## Output Files Generated

After running the script, you'll get:

1. **`submission_final.csv`** - Predictions from the best performing model
2. **`submission_ensemble.csv`** - Predictions from the ensemble model
3. **`model_performance.png`** - Visualization of model comparisons
4. **`feature_importance.png`** - Feature importance analysis plot

## Troubleshooting

### Common Issues:

1. **Missing dependencies**: Install using `pip install -r requirements.txt`
2. **File not found**: Ensure `train.csv` and `test.csv` are in the same directory
3. **Memory issues**: Reduce model complexity or use smaller datasets
4. **Plot display issues**: The script saves plots as PNG files automatically

## Contributing

Feel free to improve the code by:
- Adding more models
- Implementing feature engineering
- Adding hyperparameter tuning
- Improving visualizations
- Adding more evaluation metrics

## License

This project is open source and available under the MIT License.
