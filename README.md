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

## Technologies
- Language: Python
- Libraries: Pandas (Data Manipulation), Matplotlib & Seaborn (Data Visualization)
- Data PreProcessing: StandardScaler, LabelEncoder
- Machine Learning: Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Trees, KNN, XGBoost

## Data Cleaning

1. Checking for Missing Value:  Only one column contained missing values, **bmi** had **201 null entries** out of 5,110 rows. All other columns were complete with no nulls.
2. Dropping the ID Column: The **id** column is a unique patient identifier with no predictive value. It was dropped to prevent it from influencing the model.
3. There is no duplicate rows were found in the dataset.
4. The missing **bmi** values were filled using the **median** which is **28.1**. The median was chosen because it is least affected by outliers, which were present in the bmi distribution. After filling, the column had zero missing  values.
5. Only **1 row** had a gender value of **'Other'**. This record was removed as it is insignificant and could introduce noise during encoding

## Exploratory Data Analysis
1. Age Distribution
- The age distribution shows that stroke cases are more common among older individuals, specifically among individuals aged 60 and above and while younger individuals have very few stroke cases.
- The likelihood of having a stroke increases with age. Age is an important predictive feature for the stroke risk.
2. Average Glucose Level
- The chart suggests that individuals who had a stroke generally have higher average glucose levels (often above 200 mg/dL) compared to those who did not. This indicates a possible link between elevated blood sugar levels and stroke occurrence.
3. BMI
- Most patients fall within a BMI range of 20 to 40, but individuals with stroke tend to have slightly higher BMI values (overweight or obese) than those without stroke.
- Body mass index may contribute to stroke risk, as obesity is linked to several health conditions such as hypertension and heart disease. But BMI alone may not be a strong predictor but can still contribute when combined with other factors.

  <img width="1293" height="392" alt="AGE" src="https://github.com/user-attachments/assets/c3fb0383-9d42-4440-b038-5a4fffba506d" />
4. Hypertension
- Most individuals did not suffer from hypertension and individuals with hypertension have a higher risk of stroke cases compared to those without hypertension.
- Hypertension is a significant health risk factor and an important variable for predicting stroke.
5. Heart Disease
- Individuals with heart disease appear to have a higher chance of having a stroke compared to those without heart disease. This is due to the fact that cardiovascular diseases are closely linked to stroke risk.
- Heart Disease is a major indicator of having a stroke risk alongside other factors like age, it becomes clear that older individuals with heart disease represent one of the highest-risk categories
6. Gender
- The gender distribution shows that both males and females experience strokes, but the difference between genders is not that large.
- Gender alone is not a strong predictor of having a stroke as age or hypertension.
<img width="1279" height="340" alt="2" src="https://github.com/user-attachments/assets/d826ae2c-3f6d-4e6c-94a9-f7d77ba751ad" />

