#!/usr/bin/env python3
"""
Data Download Script for Student Dropout Prediction Project
==========================================================

This script helps users download the required data files for the project.
Since the CSV files are large, they are not included in the GitHub repository.
"""

import os
import requests
import zipfile
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✓ Downloaded {filename}")

def main():
    print("="*60)
    print("STUDENT DROPOUT PREDICTION - DATA DOWNLOAD")
    print("="*60)
    
    # Check if data files already exist
    if os.path.exists('train.csv') and os.path.exists('test.csv'):
        print("✓ Data files already exist!")
        print("You can now run: python final_dropout_prediction.py")
        return
    
    print("\n📁 Data files not found. Please download them manually:")
    print("\n1. Download the dataset from Kaggle:")
    print("   - Go to: https://www.kaggle.com/datasets/akshaydattatraykhare/student-dropout-prediction")
    print("   - Download the dataset")
    print("   - Extract train.csv and test.csv to this directory")
    
    print("\n2. Alternative: Use sample data")
    print("   - Create sample data files for testing")
    
    choice = input("\nWould you like to create sample data files for testing? (y/n): ")
    
    if choice.lower() == 'y':
        create_sample_data()
    else:
        print("\nPlease download the data files manually and place them in this directory.")
        print("Then run: python final_dropout_prediction.py")

def create_sample_data():
    """Create sample data files for testing"""
    import pandas as pd
    import numpy as np
    
    print("\nCreating sample data files...")
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample features
    features = {}
    for i in range(41):
        features[f'v_{i}'] = np.random.randn(n_samples)
    
    # Create target variable with some correlation
    features['label'] = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.5, 0.2])
    features['id'] = range(n_samples)
    
    # Create DataFrame
    train_df = pd.DataFrame(features)
    train_df = train_df[['id'] + [f'v_{i}' for i in range(41)] + ['label']]
    
    # Create sample test data
    test_df = train_df.drop('label', axis=1).copy()
    test_df['id'] = range(n_samples, n_samples + 500)
    
    # Save files
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    print("✓ Created sample data files:")
    print(f"  - train.csv ({len(train_df)} samples)")
    print(f"  - test.csv ({len(test_df)} samples)")
    print("\n⚠️  Note: These are sample files for testing only.")
    print("   For real results, download the actual dataset from Kaggle.")
    print("\nYou can now run: python final_dropout_prediction.py")

if __name__ == "__main__":
    main()
