---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: Medical
  difficulty: Middle
  module: pandas
  idx: 4
---


### Step 1. Import required python packages.
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
```

### Step 2. Load the dataset from the URL into a pandas DataFrame named "df".

```python
df = pd.read_excel("pandas\pandas_dataset04.xlsx")
```

### Step 3. Calculate and display the number of missing values in each column of the DataFrame "df".
```python
print(df.isnull().sum())
```

### Step 4. Replace missing values in the "Weight" column with the median value of the same column. Make sure to apply this change to the DataFrame "df" itself.
```python
df['Weight'].fillna(df['Weight'].median(), inplace=True)
```
 
### Step 5. Remove any remaining rows with missing values from the DataFrame "df". Ensure that changes are made inplace.
```python
df.dropna(inplace=True)
```

### Step 6. Perform square root transformation to "Height" and give the data distribution of the column.
```python
df['Height'] = np.sqrt(df['Height'])
sns.displot(df['Height'], kde=True)
```

### Step 7. Detect whether there are outliers in "Blood Pressure", set Outlier_degree = 2.5. 
```python
z_scores = zscore(df['Blood Pressure'])
abs_z_scores = np.abs(z_scores)
outlier_indices = np.where(abs_z_scores > 2.5)[0]
print(f"Outlier indices for 'Blood Pressure': {outlier_indices}")
```

### Step 8. Visualize Outliers Using Box Plots for "Blood Pressure", using Plot_Title = "Boxplot for Blood Pressure", X_axis_Label = "Blood Pressure", Y_axis_Label = "Frequency", Figure_size = (10, 6), Color = "red".
```python
plt.figure(figsize=(10,6))
sns.boxplot(data=df['Blood Pressure'], color='red')
plt.title('Boxplot for Blood Pressure')
plt.xlabel('Blood Pressure')
plt.ylabel('Frequency')
plt.show()
```

### Step 9. Convert values in the "Health Status" column to numerical values using label encoding.
```python
le = LabelEncoder()
df['Health Status'] = le.fit_transform(df['Health Status'])
```

### Step 10. Group and Aggregate Data by "Health Status" and calculate the min, max average of each group.
```python
grouped_df = df.groupby('Health Status').agg(['min', 'max', 'mean'])
print(grouped_df)
```

### Step 11. Display the correlation matrix of "Weight", "Height" and "Blood Sugar".
```python
correlation_matrix = df[['Weight', 'Height', 'Blood Sugar']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11. Display the heatmap with annotations for each cell.
```python
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```