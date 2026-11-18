Task 3 – Feature Engineering & Preprocessing Pipeline

Project Overview

This project uses the UCI Credit Card Default Dataset to perform data preprocessing, feature engineering and machine learning classification using Python.

Dataset Information

- Dataset: Credit Card Default Dataset
- Total Records: 30,000
- Input Features: 23
- Target: Default Payment (Y)

Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

Preprocessing Techniques

- Train-Test Split (80:20)
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer
- Scikit-learn Pipeline

Machine Learning Model

Random Forest Classifier

Model Evaluation

- Training Records: 24,000
- Testing Records: 6,000
- Accuracy: 81.05%

Feature Importance Analysis

The top 15 important features were identified using Random Forest feature importance.

- Highest importance: X6 (0.098390)
- Second highest: X5 (0.064849)
- Third highest: X1 (0.059048)

A bar chart was generated to visualize feature importance.

Project Files

- "Preprocessing-Pipeline.py" – Preprocessing and model training code
- "Preprocessing-Output.txt" – Actual preprocessing results
- "Feature-Importance.py" – Feature importance analysis code
- "Feature-Importance-Output.txt" – Feature importance results
- "README.md" – Project documentation

Conclusion

The preprocessing pipeline was successfully implemented, and a Random Forest Classifier achieved 81.05% accuracy. Feature importance analysis was also performed to identify the most influential input features.
