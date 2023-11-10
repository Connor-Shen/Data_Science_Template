---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: Financial
  difficulty: Middle
  module: pandas
  idx: 5
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
df = pd.read_csv("pandas\pandas_dataset05.xlsx")
```

### Step 3. Calculate and display the number of missing values in each column of the DataFrame "df".
```python
missing_values = df.isnull().sum()
print(missing_values)
```

### Step 4. Replace missing values in the "Monthly Income" column with the median value of the same column. Make sure to apply this change to the DataFrame "df" itself.
```python
df['Monthly Income'].fillna(df['Monthly Income'].median(), inplace=True)
```

### Step 5. Remove any remaining rows with missing values from the DataFrame "Balance". Ensure that changes are made inplace.
```python
df.dropna(subset=['Balance'], inplace=True)
```

### Step 6. Perform square root transformation to "Monthly Income" and give the data distribution of the column.
```python
df['Monthly Income'] = np.sqrt(df['Monthly Income'])
plt.hist(df['Monthly Income'], bins=30)
plt.xlabel('Square Root Transformed Monthly Income')
plt.ylabel('Frequency')
plt.show()
```

### Step 7. Detect whether there are outliers in "Debt", set Outlier_degree = 2.5.
```python
Q1 = df['Debt'].quantile(0.25)
Q3 = df['Debt'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Debt'] < Q1 - 2.5*IQR) | (df['Debt'] > Q3 + 2.5*IQR)]
print(outliers)
```

### Step 8. Visualize Outliers Using Box Plots for "Debt", using Plot_Title = "Boxplot for Debt", X_axis_Label = "Debt", Y_axis_Label = "Frequency", Figure_size = (10, 6), Color = "red".
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['Debt'], color="red")
plt.title("Boxplot for Debt")
plt.xlabel("Debt")
plt.ylabel("Frequency")
plt.show()
```

### Step 9. Convert values in the "Credit Rating" column to numerical values using label encoding.
```python
encoder = LabelEncoder()
df['Credit Rating'] = encoder.fit_transform(df['Credit Rating'])
```

### Step 10. Group and Aggregate Data by "Credit Rating" and calculate the min, max, average of each group.
```python
grouped = df.groupby('Credit Rating').agg(['min', 'max', 'mean'])
print(grouped)
```

### Step 11. Display the correlation matrix of "Monthly Income", "Balance", and "Debt".
```python
correlation_matrix = df[['Monthly Income', 'Balance', 'Debt']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11. Display the heatmap with annotations for each cell.
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```
