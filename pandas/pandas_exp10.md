---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: Penguins dataset
  difficulty: Hard
  module: pandas
  idx: 10
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
df = pd.read_csv("pandas\pandas_dataset10.csv")
```

### Step 3. Calculate and display the number of missing values in each column of the DataFrame "df".
```python
missing_values = df.isnull().sum()
print(missing_values)
```

### Step 4. Replace missing values in the "Culmen Length (mm)" column with the median value of the same column. Make sure to apply this change to the DataFrame "df" itself.
```python
df['Culmen Length (mm)'].fillna(df['Culmen Length (mm)'].median(), inplace=True)
```

### Step 5. Remove any remaining rows with missing values from the DataFrame "Culmen Depth (mm)". Ensure that changes are made inplace.
```python
df.dropna(subset=['Culmen Depth (mm)'], inplace=True)
```

### Step 6. Perform square root transformation to "Body Mass (g)" and give the data distribution of the column.
```python
df['Body Mass (g)'] = np.sqrt(df['Body Mass (g)'])
plt.hist(df['Body Mass (g)'], bins=30)
plt.xlabel('Square Root Transformed Body Mass (g)')
plt.ylabel('Frequency')
plt.show()
```

### Step 7. Detect whether there are outliers in "Culmen Length (mm)", set Outlier_degree = 2.5.
```python
Q1 = df['Culmen Length (mm)'].quantile(0.25)
Q3 = df['Culmen Length (mm)'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Culmen Length (mm)'] < Q1 - 2.5*IQR) | (df['Culmen Length (mm)'] > Q3 + 2.5*IQR)]
print(outliers)
```

### Step 8. Visualize Outliers Using Box Plots for "Culmen Length (mm)", using Plot_Title = "Boxplot for Culmen Length (mm)", X_axis_Label = "Culmen Length (mm)", Y_axis_Label = "Frequency", Figure_size = (10, 6), Color = "green".
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['Culmen Length (mm)'], color="green")
plt.title("Boxplot for Culmen Length (mm)")
plt.xlabel("Culmen Length (mm)")
plt.ylabel("Frequency")
plt.show()
```

### Step 9. Convert values in the "Species" column to numerical values using label encoding.
```python
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])
```

### Step 10. Group and Aggregate Data by "Species" and calculate the min, max, average of each numerical group.
```python
grouped = df.groupby('Species').agg(['min', 'max', 'mean'])
print(grouped)
```

### Step 11. Display the correlation matrix of "Culmen Length (mm)", "Culmen Depth (mm)", and "Flipper Length (mm)".
```python
correlation_matrix = df[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11. Display the heatmap with annotations for each cell.
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```
