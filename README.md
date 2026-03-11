# Capstone-Project-Stroke-Prediction

## Project Overview
According to the World Health Organization (WHO), Stroke is the 3rd leading cause of death responsible for approximately 10% of total deaths and long-term disability worldwide.

This project aims to predict whether a patient is likely to get a stroke based on input parameters like gender, age, bmi, marital status, various diseases, and smoking status. The goal is to build a classification model that can identify individuals who are at high risk of stroke based on various health indicators.

The project includes data cleaning, exploratory data analysis (EDA), feature preprocessing, model training, tuning and evaluation of multiple classification algorithms.

## Dataset
The dataset was gotten from an open online data source called Kaggle. [Download Here](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

1. Rows: 5,110 patients record
2. Target: Stroke (1 if the patient had a stroke, 0 if not).
 
   | Feature           | Description                                 |
   | ----------------- | ------------------------------------------- |
   | age               | Age of the patient                          |
   | gender            | Male or Female                              |
   | hypertension      | Whether the patient has high blood pressure (1 = Yes, 0 = No)|
   | heart_disease     | Presence of heart disease (1 = Yes, 0 = No)                   |
   | ever_married      | Whether the patient is married              |
   | work_type         | Employment type                             |
   | Residence_type    | Urban or Rural residence                    |
   | avg_glucose_level | Average blood glucose level                 |
   | bmi               | Body Mass Index of the patient              |
   | smoking_status    | Smoking behavior of the patient              |
   | stroke            |  Target variable indicating whether a stroke occurred (1 = Yes, 0 = No) |

  ## Data Loading

  1. All necessary libraries were imported at the start of the notebook, covering data manipulation, exploratory data analysis, visualization, machine learning, and class imbalance handling.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_auc_score,roc_curve
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")
```

2. Loading Dataset
```python
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.head()
```
The dataset contains **5,110 rows and 12 columns**, including patients' gender, age, bmi, average glucose level, marital status, employment status, lifestyle, and medical information.

3. 
https://github.com/Bembem11/Capstone-Project-Stroke-Prediction/blob/main/MyCapstoneProject.ipynb#cell-3
