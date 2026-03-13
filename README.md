# Capstone-Project-Stroke-Prediction

## Project Overview
According to the World Health Organization (WHO), Stroke is the 3rd leading cause of death responsible for approximately 10% of total deaths and long-term disability worldwide.

This project aims to predict whether a patient is likely to get a stroke based on input parameters like gender, age, bmi, marital status, various diseases, and smoking status. The goal is to build a classification model that can identify individuals who are at high risk of stroke based on various health indicators.

The project includes data cleaning, exploratory data analysis (EDA), feature preprocessing, model training, tuning and evaluation of multiple classification algorithms.

## Dataset
The dataset was gotten from an open online data source called Kaggle. [Download Here](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Key Features:
* Demographics: gender, age, ever_married, work_type, Residence_type
* Health Metrics: bmi (Body Mass Index), avg_glucose_level
* Medical History: hypertension, heart_disease, smoking_status
* Target Variable: stroke (1 = Patient suffered a stroke, 0 = No stroke)
  
  *Note: Missing values in the bmi column were handled using median imputation to maintain data integrity.*
 
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
- Data Manipulation: Pandas, Numpy
- Data Visualization: Matplotlib & Seaborn 
- Data PreProcessing: StandardScaler, LabelEncoder, SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance.
- Machine Learning: Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Trees, KNN, XGBoost

## Data Cleaning

1. Checking for Missing Value:  Only one column contained missing values, **bmi** had **201 null entries** out of 5,110 rows. All other columns were complete with no nulls.
2. Dropping the ID Column: The **id** column is a unique patient identifier with no predictive value. It was dropped to prevent it from influencing the model.
3. There is no duplicate rows were found in the dataset.
4. The missing **bmi** values were filled using the **median** which is **28.1**. The median was chosen because it is least affected by outliers, which were present in the bmi distribution. After filling, the column had zero missing  values.
5. Only **1 row** had a gender value of **'Other'**. This record was removed as it is insignificant and could introduce noise during encoding

## Exploratory Data Analysis
Before training predictive models, an exploratory analysis was conducted to uncover the underlying distributions and relationships between patient attributes and stroke occurrence. 

### 1. Age Distribution Analysis
* *Observation:* The density plot analysis illustrates that stroke occurrences are predominantly concentrated in the older demographic, specifically between 60 and 85 years of age. Conversely, the non-stroke population is distributed across a much wider and younger range, with a higher density between 30 and 60 years.
* *Key Insight:* There is a strong positive relationship between advancing age and stroke risk. Age acts as a primary baseline indicator and is expected to be a highly influential predictor in the classification models.

### 2. Average Glucose Level Distribution
* *Observation:* There is a distinct divergence in average glucose levels between the two groups. Patients without a stroke history typically cluster around healthy baseline levels of 80–120 mg/dL. However, the stroke group tends to exhibit higher glucose levels, marked by a secondary density peak around 200 mg/dL.
* *Key Insight:* Elevated glucose levels are a significant risk marker. This distribution suggests that underlying metabolic conditions, such as diabetes or pre-diabetes, heavily compound the risk of a stroke.

### 3. The Nuance of Body Mass Index (BMI)
* *Observation:* The isolated BMI distributions for both stroke and non-stroke groups display significant overlap, with the vast majority of observations concentrated between 25 and 35. While stroke cases lean slightly toward higher BMI values, the visual distinction between the two classes is not sharply pronounced.
* *Key Insight:* When evaluated independently during EDA, BMI does not strongly differentiate the two classes, suggesting limited predictive power in isolation. (Note: As detailed in the Feature Importance section, advanced tree-based models later identified BMI as a top predictor, indicating that its interaction with other variables like Age and Glucose is highly significant, even if it doesn't separate the classes linearly).

  <img width="1293" height="392" alt="AGE" src="https://github.com/user-attachments/assets/c3fb0383-9d42-4440-b038-5a4fffba506d" />
### 4. Hypertension
* *Observation:* The majority of individuals in the dataset do not suffer from hypertension. However, when isolating stroke outcomes, those with hypertension exhibit a notably higher proportion of stroke cases compared to the non-hypertensive group.
* *Key Insight:* There is a clear positive association between hypertension and stroke occurrence, aligning with established medical consensus that high blood pressure is a major, direct risk factor.

### 5. Heart Disease
* *Observation:* Similar to hypertension, most individuals in the dataset do not have a history of heart disease. Nevertheless, the proportion of stroke cases is significantly higher among those who do. 
* *Key Insight:* Pre-existing heart disease substantially increases the likelihood of a stroke, making it a critical baseline medical feature for predictive modeling.

### 6. Gender
* *Observation:* The dataset contains a higher number of female participants than male participants. However, stroke cases are distributed across both genders in relatively similar proportions.
* *Key Insight:* Gender does not exhibit a strong direct association with stroke occurrence in this dataset, indicating it will likely have a weaker predictive contribution compared to cardiovascular indicators.
<img width="1279" height="340" alt="2" src="https://github.com/user-attachments/assets/d826ae2c-3f6d-4e6c-94a9-f7d77ba751ad" />

### 7. Smoking Status
* *Observation:* The majority of the dataset falls into the "never smoked" or "unknown" categories. Interestingly, a relatively higher proportion of stroke cases appears among individuals who "formerly smoked" compared to current smokers—likely because individuals often quit smoking after developing preliminary health complications. The "unknown" category introduces some noise into the dataset.
* *Key Insight:* Smoking behavior contributes to stroke risk, but its effect in this dataset is moderate when compared to dominant medical factors like age, glucose levels, and heart disease. 

### 8. Work Type
* *Observation:* The "private sector" category dominates the dataset, followed by "self-employed" and "government" roles. Categories like "children" and "never worked" contain almost zero stroke cases. 
* *Key Insight:* Work type itself is not a direct biological cause of stroke. Instead, it acts as a proxy variable for age and socioeconomic status. For example, the "children" category naturally represents a younger demographic with a near-zero stroke probability, while "private" and "self-employed" represent the aging adult workforce.

### 9. Residence Type
* *Observation:* The dataset shows a nearly 50/50 split between urban and rural residents. Stroke cases are also distributed very evenly between these two environments, with urban areas showing only a negligible increase.
* *Key Insight:* Residence type (urban vs. rural) demonstrates limited predictive power for stroke occurrence and is overshadowed by specific health and lifestyle metrics.
  
<img width="1272" height="348" alt="smoking status" src="https://github.com/user-attachments/assets/385b0155-70bd-4237-9d0b-972a3d747999" />



## Summary based on the EDAs
### BMI Distribution
* *Continuous Distribution:* Initial observation of the continuous BMI distributions showed significant overlap between the two groups (mostly concentrated between 25 and 35), making it difficult to sharply differentiate stroke and non-stroke cases at a glance.
 * *Categorical Breakdown:* However, when segmented into standard BMI categories, a stark contrast emerges:
  * *Stroke Cases:* Overwhelmingly concentrated in higher BMI brackets. *Overweight (46.2%)* and *Obese (39.4%)* make up a combined *85.6%* of all stroke cases. Normal weight (14.1%) and Underweight (0.4%) are exceptionally rare.
  * *Non-Stroke Cases:* Exhibit a broader, more balanced spread. While Obese (37.5%) and Overweight (30.8%) are still prominent, there is a much larger share of Normal weight (24.8%) and Underweight (6.9%) individuals.
  *  *Key Insight:* Categorical analysis reveals that elevated BMI (specifically in the overweight and obese ranges) is strongly linked to increased stroke occurrence. This granular finding perfectly contextualizes why advanced machine learning models later identified BMI as the single most critical predictor of stroke risk in this dataset.

<img width="695" height="514" alt="bmi" src="https://github.com/user-attachments/assets/55b45cdc-a203-4ca5-87f7-849f94c7179b" />

### Age Distribution
* Stroke probability scales dramatically across age brackets. The risk is almost negligible (<0.5%) in patients under 40, climbs to ~13% in the 60–80 demographic, and peaks at nearly 20% for those over 80.
* The likelihood of a stroke accelerates significantly after age 40 and becomes highly prevalent after 60, cementing age as the strongest non-modifiable baseline predictor in the dataset.

### Hypertension & Heart Disease
*  An intersection analysis of underlying conditions reveals significant overlap. While hypertension is the most common standalone condition (381 cases), it frequently compounds with others: 53 patients had both hypertension and a stroke, 34 had heart disease and a stroke, and 13 patients suffered from all three conditions simultaneously.
*  Hypertension acts as a major intersecting risk factor that heavily amplifies the likelihood of developing heart disease, suffering a stroke, or both.






### Machine Learning Models Used
Several classification algorithms were trained and compared, including:

Logistic Regression
Random Forest
Gradient Boosting
Decision Trees
Each model was evaluated using train-test splitting and cross-validation to ensure reliable performance.







## Models Performance

Seven different classification models were trained and tested. Ensemble methods significantly outperformed traditional linear models for this dataset.

| Model | Accuracy Score | 
| :--- | :--- |
| *XGBClassifier* | *0.93* |
| *RandomForestClassifier* | *0.92* |
| GradientBoostingClassifier | *0.91* |
| DecisionTreeClassifier | *0.89* |
| KNeighborsClassifier | *0.83* |
| SVC | *0.78* |
| LogisticRegression | *0.74* |

*Top Performer:* The XGBClassifier achieved the highest accuracy at *93%, closely followed by the RandomForestClassifier at **92%*. These models demonstrate strong predictive capability and are highly suited for this diagnostic classification task.

 
 ## Key Findings & Feature Importance
 To make the model interpretable for healthcare professionals, feature importance was extracted. The analysis revealed that demographic and metabolic factors are the strongest predictors of stroke risk.

Top 3 Stroke Indicators:

* *BMI (0.291)**: Body Mass Index emerged as the most significant predictor in the dataset.

* *Age (0.219)**: Advancing age is the second most critical risk factor.

* *Average Glucose Level (0.211)**: Blood sugar levels strongly correlate with stroke likelihood.

*Secondary Factors*: Smoking status (0.078), Residence type (0.054), and Work type (0.043) also contributed to the model, while baseline conditions like existing heart disease (0.026) and hypertension (0.031) had a surprisingly lower relative weight in this specific dataset's tree-based splits.

## Early Prevention Strategies
By identifying **BMI**, **Age**, and **Glucose Levels** as the primary drivers, this model can be integrated into early-warning health systems and faster decision-making.

* Early Risk Screening: Identify high-risk patients through regular health checkups and predictive models.
* Blood Pressure Control: Monitor and manage hypertension through medication and lifestyle adjustments.
* Diabetes & Glucose Management: Encourage regular blood sugar monitoring and healthy diet plans.
* Lifestyle Improvement Programs: Promote smoking cessation, exercise, healthy diet, and weight management.
* Personalized Preventive Care: Provide targeted monitoring and treatment plans for patients identified as high risk.

Clinicians and preventative care platforms can use these insights to flag high-risk patients for targeted interventions, lifestyle coaching, or further medical screening before a critical event occurs.

