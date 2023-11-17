---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: Education
  difficulty: Middle
  module: pandas
  idx: 6
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

df = pd.read_excel("pandas\pandas_dataset06.xlsx")
```

### Step 3. Calculate and display the number of missing values in each column of the DataFrame "df".
```python
missing_values = df.isnull().sum()
print(missing_values)
```

### Step 4. Replace missing values in the "Exam Score" column with the median value of the same column. Make sure to apply this change to the DataFrame "df" itself.
```python
df['Exam Score'].fillna(df['Exam Score'].median(), inplace=True)
```

### Step 5. Remove any remaining rows with missing values from the DataFrame "Attendance". Ensure that changes are made inplace.
```python
df.dropna(subset=['Attendance'], inplace=True)
```

### Step 6. Perform square root transformation to "Exam Score" and give the data distribution of the column.
```python
df['Exam Score'] = np.sqrt(df['Exam Score'])
plt.hist(df['Exam Score'], bins=30)
plt.xlabel('Square Root Transformed Exam Score')
plt.ylabel('Frequency')
plt.show()
```

### Step 7. Detect whether there are outliers in "Online Exam Count", set Outlier_degree = 2.5.
```python
Q1 = df['Online Exam Count'].quantile(0.25)
Q3 = df['Online Exam Count'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Online Exam Count'] < Q1 - 2.5*IQR) | (df['Online Exam Count'] > Q3 + 2.5*IQR)]
print(outliers)
```

### Step 8. Visualize Outliers Using Box Plots for "Online Exam Count", using Plot_Title = "Boxplot for Online Exam Count", X_axis_Label = "Online Exam Count", Y_axis_Label = "Frequency", Figure_size = (10, 6), Color = "green".
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['Online Exam Count'], color="green")
plt.title("Boxplot for Online Exam Count")
plt.xlabel("Online Exam Count")
plt.ylabel("Frequency")
plt.show()
```

### Step 9. Convert values in the "Performance" column to numerical values using label encoding.
```python
encoder = LabelEncoder()
df['Performance'] = encoder.fit_transform(df['Performance'])
```

### Step 10. Group and Aggregate Data by "Performance" and calculate the min, max, average of each group.
```python
grouped = df.groupby('Performance').agg(['min', 'max', 'mean'])
print(grouped)
```

### Step 11. Display the correlation matrix of "Exam Score", "Attendance", and "Online Exam Count".
```python
correlation_matrix = df[['Exam Score', 'Attendance', 'Online Exam Count']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11. Display the heatmap with annotations for each cell.
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```
