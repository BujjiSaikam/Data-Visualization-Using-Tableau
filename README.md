# **Financial Analytics**

### Description
This project aims to analyze financial data to understand the relationship between market capitalization and quarterly sales of companies. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, storage in an SQLite database, machine learning model development for prediction, and visualization of results.

### Table of Contents
1. [Introduction](#introduction)
2. [Tools and Technologies](#tools-and-technologies)
3. [Data Collection](#data-collection)
4. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Additional Visualizations](#additional-visualizations)
7. [Feature Engineering](#feature-engineering)
8. [Data Storage using SQL](#data-storage-using-sql)
9. [Machine Learning Model for Prediction](#machine-learning-model-for-prediction)
10. [Visualization of Results](#visualization-of-results)
11. [Conclusion](#conclusion)

### Introduction
**Overview**: This project aims to analyze financial data to understand the relationship between market capitalization and quarterly sales of companies. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, storage in an SQLite database, machine learning model development for prediction, and visualization of results.

**Objectives**: The main objective is to build a predictive model to estimate market capitalization based on quarterly sales data, providing actionable insights for financial decision-making.

### Tools and Technologies
**Programming Languages**: Python, SQL  
**Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
**Software/Platforms**: Jupyter Notebook, SQLite for data storage  

### Data Collection
**Data Sources**: The dataset includes financial data such as market capitalization and quarterly sales, sourced from a CSV file.

**Data Description**: Detailed information about columns like company name, market capitalization, and quarterly sales.

### Data Loading and Preprocessing
- **Import necessary libraries**:
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  ```

- **Load the dataset**:
  ```python
  file_path = '/content/sample_data/Financial Analytics data .csv'
  data = pd.read_csv(file_path)
  data.head()
  ```

- **Basic data preprocessing**:
  ```python
  data.dropna(inplace=True)
  data.columns = [col.strip() for col in data.columns]
  data.columns
  ```

### Exploratory Data Analysis (EDA)
- **Summary statistics, distribution plots, and correlation analysis**:
  ```python
  summary_stats = data.describe()
  summary_stats
  ```

- **Distribution plots**:
  ```python
  plt.figure(figsize=(14, 6))
  plt.subplot(1, 2, 1)
  sns.histplot(data['Mar Cap - Crore'], bins=30, color='orange')
  plt.title('Market Capitalization Distribution')

  plt.subplot(1, 2, 2)
  sns.histplot(data['Sales Qtr - Crore'], bins=30, color='red')
  plt.title('Quarterly Sales Distribution')

  plt.tight_layout()
  plt.show()
  ```

### Additional Visualizations
- **Pairplot**:
  ```python
  sns.pairplot(data[['Mar Cap - Crore', 'Sales Qtr - Crore']])
  plt.show()
  ```

- **Box Plots**:
  ```python
  plt.figure(figsize=(14, 6))
  plt.subplot(1, 2, 1)
  sns.boxplot(y=data['Mar Cap - Crore'], color='orange')
  plt.title('Market Capitalization Box Plot')

  plt.subplot(1, 2, 2)
  sns.boxplot(y=data['Sales Qtr - Crore'], color='red')
  plt.title('Quarterly Sales Box Plot')

  plt.tight_layout()
  plt.show()
  ```

### Feature Engineering
- **Adding new features (log transformations)**:
  ```python
  data['Log_Mar_Cap'] = np.log1p(data['Mar Cap - Crore'])
  data['Log_Sales_Qtr'] = np.log1p(data['Sales Qtr - Crore'])
  ```

### Data Storage using SQL
- **Store data in an SQLite database for efficient querying**:
  ```python
  import sqlite3

  # Connect to SQLite database
  conn = sqlite3.connect('financial_data.db')
  data.to_sql('companies', conn, if_exists='replace', index=False)

  # Querying the data
  query = "SELECT * FROM companies WHERE `Mar Cap - Crore` > 100000"
  df_query = pd.read_sql(query, conn)
  ```

### Machine Learning Model for Prediction
- **Linear regression model to predict market capitalization**:
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import mean_squared_error, r2_score

  # Features and target variable
  X = data[['Sales Qtr - Crore']]
  y = data['Mar Cap - Crore']

  # Train-test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Model training
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Predictions
  y_pred = model.predict(X_test)

  # Evaluation
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  ```

### Visualization of Results
- **Scatter plot of actual vs predicted values**:
  ```python
  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, y_pred, alpha=0.7)
  plt.xlabel('Actual Market Cap')
  plt.ylabel('Predicted Market Cap')
  plt.title('Actual vs Predicted Market Capitalization')
  plt.show()
  ```

### Conclusion
**Summary**: This project effectively utilized financial data analysis techniques, including EDA, SQL integration, machine learning modeling, and visualization, to predict market capitalization based on quarterly sales data.

**Implications**: Insights gained can inform strategic decisions regarding market trends and company performance.

**Future Work**: Potential future research could include exploring additional features or employing more advanced machine learning algorithms for improved prediction accuracy.

Thank you for your time!
