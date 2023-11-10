---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: Online shoppers
  difficulty: Middle
  module: pandas
  idx: 7
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
df = pd.read_csv("pandas\pandas_dataset07.csv")
```

### Step 3. Calculate and display the number of missing values in each column of the DataFrame "df".
```python
missing_values = df.isnull().sum()
print(missing_values)
```

### Step 4. Replace missing values in the "Administrative" column with the median value of the same column. Make sure to apply this change to the DataFrame "df" itself.
```python
df['Administrative'].fillna(df['Administrative'].median(), inplace=True)
```

### Step 5. Remove any remaining rows with missing values from the DataFrame "Administrative_Duration". Ensure that changes are made inplace.
```python
df.dropna(subset=['Administrative_Duration'], inplace=True)
```

### Step 6. Perform square root transformation to "BounceRates" and give the data distribution of the column.
```python
df['BounceRates'] = np.sqrt(df['BounceRates'])
plt.hist(df['BounceRates'], bins=30)
plt.xlabel('Square Root Transformed BounceRates')
plt.ylabel('Frequency')
plt.show()
```

### Step 7. Detect whether there are outliers in "ProductRelated_Duration", set Outlier_degree = 2.5.
```python
Q1 = df['ProductRelated_Duration'].quantile(0.25)
Q3 = df['ProductRelated_Duration'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['ProductRelated_Duration'] < Q1 - 2.5*IQR) | (df['ProductRelated_Duration'] > Q3 + 2.5*IQR)]
print(outliers)
```

### Step 8. Visualize Outliers Using Box Plots for "ProductRelated", using Plot_Title = "Boxplot for ProductRelated", X_axis_Label = "ProductRelated", Y_axis_Label = "Frequency", Figure_size = (10, 6), Color = "green".
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['ProductRelated'], color="green")
plt.title("Boxplot for ProductRelated")
plt.xlabel("ProductRelated")
plt.ylabel("Frequency")
plt.show()
```

### Step 9. Convert values in the "VisitorType" column to numerical values using label encoding.
```python
encoder = LabelEncoder()
df['VisitorType'] = encoder.fit_transform(df['VisitorType'])
```

### Step 10. Group and Aggregate Data by "VisitorType" and calculate the min, max, average of each numerical group.
```python
grouped = df.groupby('VisitorType').agg(['min', 'max', 'mean'])
print(grouped)
```

### Step 11. Display the correlation matrix of "ProductRelated_Duration", "BounceRates", and "ExitRates".
```python
correlation_matrix = df[['ProductRelated_Duration', 'BounceRates', 'ExitRates']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11. Display the heatmap with annotations for each cell.
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```
