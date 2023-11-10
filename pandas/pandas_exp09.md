---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: DS Salary
  difficulty: Hard
  module: pandas
  idx: 9
---

### Step 1. Import required python packages.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
```

### Step 2. Load the dataset from the URL into a pandas DataFrame named "df".
(Note: You didn't provide a specific URL for this dataset, so I'm using a placeholder. Please replace it with the actual URL.)
```python
URL = "YOUR_DATASET_URL_HERE"
df = pd.read_csv("pandas\pandas_dataset09.csv")
```

### Step 3. Calculate and display the number of missing values in each column of the DataFrame "df".
```python
missing_values = df.isnull().sum()
print(missing_values)
```

### Step 4. Replace missing values in the "remote_ratio" column with the median value of the same column. Make sure to apply this change to the DataFrame "df" itself.
```python
df['remote_ratio'].fillna(df['remote_ratio'].median(), inplace=True)
```

### Step 5. Remove any remaining rows with missing values from the DataFrame "employment_type". Ensure that changes are made inplace.
```python
df.dropna(subset=['employment_type'], inplace=True)
```

### Step 6. Perform square root transformation to "salary_in_usd" and give the data distribution of the column.
```python
df['salary_in_usd'] = np.sqrt(df['salary_in_usd'])
plt.hist(df['salary_in_usd'], bins=30)
plt.xlabel('Square Root Transformed salary_in_usd')
plt.ylabel('Frequency')
plt.show()
```

### Step 7. Detect whether there are outliers in "salary", set Outlier_degree = 2.5.
```python
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['salary'] < Q1 - 2.5*IQR) | (df['salary'] > Q3 + 2.5*IQR)]
print(outliers)
```

### Step 8. Visualize Outliers Using Box Plots for "salary", using Plot_Title = "Boxplot for salary", X_axis_Label = "salary", Y_axis_Label = "Frequency", Figure_size = (10, 6), Color = "green".
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['salary'], color="green")
plt.title("Boxplot for salary")
plt.xlabel("salary")
plt.ylabel("Frequency")
plt.show()
```

### Step 9. Convert values in the "job_title" column to numerical values using label encoding.
```python
encoder = LabelEncoder()
df['job_title'] = encoder.fit_transform(df['job_title'])
```

### Step 10. Group and Aggregate Data by "job_title" and calculate the min, max, average of each numerical group.
```python
grouped = df.groupby('job_title').agg(['min', 'max', 'mean'])
print(grouped)
```

### Step 11. Display the correlation matrix of "job_title", "salary", and "remote_ratio".
```python
correlation_matrix = df[['job_title', 'salary', 'remote_ratio']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11. Display the heatmap with annotations for each cell.
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```
