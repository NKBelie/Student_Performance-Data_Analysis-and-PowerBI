# Personal Introduction
**Names :** NDAYISABA KAMARIZA Belie<br>
**ID :** 27174<br>
**Course:** Introduction to Big Data Analytics<br>
**Concentration:** Software Engineering


# Student Performance Data Analysis and PowerBI
This dataset contains attributes from two Portuguese secondary schools. It includes information on students’ grades, demographic features, social and school-related features.

# Introduction

This project explores student academic performance using data from the UCI Machine Learning Repository. It aims to identify how study time, internet access, past failures, and demographic factors influence final grades. Python was used for data preprocessing and machine learning modeling, while Power BI was used to design an interactive dashboard. The insights generated help educators and stakeholders understand key performance drivers and support data-driven educational improvements.

# PROBLEM DEFINITION AND PLANNING
## I. Sector Selection
  - **Education**
## II. Problem Statement

Despite numerous educational interventions, many students continue to underperform academically, raising concerns about the underlying causes of low academic achievement. Among various factors influencing student success, the amount of time students dedicate to studying is widely considered to have a significant impact. However, the strength and nature of this relationship remain unclear in real-world educational settings.

This project aims to analyze the relationship between the number of hours students study per week (as captured by the studytime variable) and their academic performance (measured by final grades G3). Using the Student Performance dataset from the UCI Machine Learning Repository, we seek to uncover patterns, correlations, and trends that can inform students, educators, and policymakers on how study habits relate to academic success.

### Project Objectives

 - To examine how study time (per week) influences students’ final academic performance.
 - To identify trends and patterns in academic achievement based on varying levels of study time.
 - To evaluate the strength of correlation between study time and final grades.
 - To visualize the relationship using data dashboards and statistical plots.
 - To provide recommendations that encourage effective study habits for improved academic outcomes.
   
### Research Questions

 - Is there a significant relationship between the number of hours students study and their final grades?
 - Do students who study more tend to perform better academically?
 - What is the distribution of study time among students, and how does it align with performance?
 - Are there other factors (e.g., absences or failures) that moderate the relationship between study time and performance?

## III. Dataset Identification

  - **Dataset Title:** Student Performance Data Set
  - **Source Link:**[Dataset Source](https://archive.ics.uci.edu/dataset/320/student+performance)
  - **Number of Rows and Columns**:649 rows × 33 columns
  - **Data Structure:** Structured (CSV)
  - **Data Status:**  Requires Preprocessing

# PYTHON ANALYTICS TASKS

# Conduct Exploratory Data Analysis (EDA)
## Generate descriptive statistics
```
print("Math Dataset Summary:\n", mat_df.describe())
print("\n Portuguese Dataset Summary:\n", por_df.describe())
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Math%20Statistic.PNG)
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Por%20Statistics.PNG)
```
print("Math Dataset: G3 Scores by Study Time\n")
print(mat_df.groupby('studytime')['G3'].describe())

print("\n Portuguese Dataset: G3 Scores by Study Time\n")
print(por_df.groupby('studytime')['G3'].describe())
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/G3%20Scores.PNG)

## Visualize distributions and relationships among variables
```
# Visualize the relationship between study time and final grade (G3) for both datasets
plt.figure(figsize=(12, 6))
sns.histplot(mat_df['G3'], kde=True, color='blue')
plt.title("Distribution of Final Grade - Math")
plt.xlabel("Final Grade (G3)")
plt.show()
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/G3%20Math.png)
```
plt.figure(figsize=(12, 6))
sns.histplot(por_df['G3'], kde=True, color='green')
plt.title("Distribution of Final Grade - Portuguese")
plt.xlabel("Final Grade (G3)")
plt.show()
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/G3%20Por.png)

## Study Time Distribution
```
mat_df['subject'] = 'Math'
por_df['subject'] = 'Portuguese'
combined_df = pd.concat([mat_df, por_df], ignore_index=True)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.countplot(x='studytime', hue='subject', data=combined_df)

plt.title("Study Time Distribution: Math vs Portuguese")
plt.xlabel("Study Time Level")
plt.ylabel("Number of Students")
plt.legend(title='Subject')
plt.tight_layout()
plt.show()
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Study%20Time.png)

## Scatter Plot: Study Time vs Final Grade
```
sns.boxplot(x='studytime', y='G3', data=mat_df)
plt.title("Math: Final Grade by Study Time Level")
plt.xlabel("Study Time Level")
plt.ylabel("Final Grade")
plt.show()
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Math%20Final.png)
```
sns.boxplot(x='studytime', y='G3', data=por_df, color="darkblue")
plt.title("Portuguese: Final Grade by Study Time Level")
plt.xlabel("Study Time Level")
plt.ylabel("Final Grade")
plt.show()
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/por%20Final.png)

## Heatmap: Correlation Matrix
```
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
# Create a heatmap with annotations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap - Portuguese Dataset")
plt.show()
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Heatmap%20Por.png)
```
# Plot correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")

plt.title("Correlation Heatmap - Math Dataset")
plt.show()
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Heatmap%20Math.png)

# Apply a Machine Learning or Clustering Model
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis(dataset_path, subject_name):
    print(f"\n--- ANALYSIS FOR {subject_name.upper()} ---")

    # Load dataset
    df = pd.read_csv(dataset_path, sep=';')

    # Features and target for regression
    features = ['studytime', 'failures', 'absences', 'G1', 'G2']
    X = df[features]
    y = df['G3']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train regression model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = reg.predict(X_test)
    print(f"Regression Model Performance ({subject_name}):")
    print(f" - Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    print(f" - R-squared: {r2_score(y_test, y_pred):.2f}")

    # Coefficients
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': reg.coef_})
    print("Regression Coefficients:")
    print(coef_df)

    # Clustering
    cluster_features = ['studytime', 'G1', 'G2', 'G3']
    X_cluster = df[cluster_features]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_cluster)

    # Plot clusters
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='studytime', y='G3', hue='cluster', palette='viridis', s=60)
    plt.title(f'Clusters by Study Time and Final Grade ({subject_name})')
    plt.xlabel('Study Time')
    plt.ylabel('Final Grade (G3)')
    plt.legend(title='Cluster')
    plt.show()

# Run for Portuguese
run_analysis('student-por.csv', 'Portuguese')

# Run for Mathematics
run_analysis('student-mat.csv', 'Mathematics')
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Analyse%20Por.PNG)
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Clusters%20Por.png)
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Analyse%20Math.PNG)
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Clusters%20Math.png)
```
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Assuming you already have y_test and y_pred from your trained model

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
```
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Model.PNG)

# POWER BI DASHBOARD TASKS
##  Problem and Insights
This project explores the link between students’ study time and academic performance using the UCI Student Performance dataset. Analysis shows that students who study more tend to score higher in their final grades. Prior grades (G1, G2), internet access, and past failures also play important roles. A machine learning model (Random Forest) predicted final scores with high accuracy (R² ≈ 0.85), and clustering revealed clear student performance groups. These findings can help schools support students based on study habits and risk factors.

## G3 Avarage By Study Time and Schools

![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/1.PNG)

## G1,2,3 Avarage by Sex
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/2.PNG)

## Study Time Sum and G3 By G3

![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/3.PNG)

## Failure Avarage by Internet and Travel Time
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/4.PNG)

## Higher Count by School and Sex
![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/6.PNG)

## Student Performance Dashboard

This Power BI dashboard analyzes student performance across demographics and study habits. It includes interactive filters for sex and school, summary cards showing average grades, and visuals comparing study time, internet access, failures, and higher education aspirations. The report helps identify patterns affecting final grades and supports data-driven decisions to improve student outcomes.

![](https://github.com/NKBelie/Student_Performance-Data_Analysis-and-PowerBI/blob/main/Image/Dashboard.PNG)

# Conclusion

This project successfully combined Python analytics and Power BI visualization to explore factors influencing student academic performance. The analysis revealed that study time, internet access, past failures, and future education aspirations are key drivers of final grades.

By applying machine learning models and creating an interactive dashboard, we transformed raw student data into actionable insights. These findings can help educators, policymakers, and institutions implement targeted strategies to support student success and improve learning outcomes.

# Recommendations

Based on the analysis, these actions are recommended to improve student outcomes:
 - Encourage consistent study habits: Students with higher study time achieved better grades; structured study plans can help raise performance.
 - Provide targeted support to students with past failures: Focused tutoring or mentoring can help close performance gaps.
 - Increase internet access and digital learning resources: Access to online materials correlated with higher achievement.
 - Promote higher education aspirations: Students who planned to continue education performed better, suggesting that career guidance could boost motivation.
 - Monitor demographic disparities: Schools and educators should track performance differences by gender and school to ensure equity.

#  Future Work

Potential extensions of this project include:
 - Expand analysis to the Math dataset for comparison with Portuguese performance.
 - Incorporate additional socioeconomic variables (e.g., parental education, household income) to enrich insights.
 - Develop advanced predictive models, such as Gradient Boosting or Neural Networks, to improve grade prediction accuracy.
 - Deploy an online interactive dashboard accessible to educators and stakeholders.
 - Integrate R visuals and What-If Analysis in Power BI to explore interventions and their potential impact.



